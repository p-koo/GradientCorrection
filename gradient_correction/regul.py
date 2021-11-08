import numpy as np
import tensorflow as tf
from tensorflow import keras
import tfomics
from tfomics.fit import MonitorMetrics, LRDecay, EarlyStopping, progress_bar

@tf.function
def saliency_map(X, model):
  with tf.GradientTape() as tape:
    tape.watch(X)
    outputs = model(X, training=True)[:,0]
  return tape.gradient(outputs, X)

@tf.function
def calculate_angles(saliency_score):
  orthogonal_residual = tf.reduce_sum(saliency_score, axis=-1)
  L2_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(saliency_score), axis=-1))
  sine = 1/2 * orthogonal_residual / L2_norm 
  sine = tf.math.asin(sine) * tf.constant(180/3.1416) 
  return sine


class CustomTrainer():
  """Custom training loop from scratch"""

  def __init__(self, model, loss, optimizer, metrics, reg_factor=1e-4):
    self.model = model
    self.loss = loss
    self.optimizer = optimizer
    self.reg_factor = tf.constant(reg_factor)

    # metrics to monitor
    metric_names = []
    for metric in metrics:
        metric_names.append(metric)

    # class to help monitor metrics
    self.metrics = {}
    self.metrics['train'] = MonitorMetrics(metric_names, 'train')
    self.metrics['valid'] = MonitorMetrics(metric_names, 'valid')
    self.metrics['test'] = MonitorMetrics(metric_names, 'test')

  @tf.function
  def train_step(self, x, y, metrics):
    """training step for a mini-batch"""
    
    with tf.GradientTape() as tape:
      preds = self.model(x, training=True)
      loss = self.loss(y, preds)
      saliency = saliency_map(x, self.model)
      sine = calculate_angles(saliency)
      l2_norm = tf.math.sqrt(tf.reduce_sum(tf.math.square(sine), axis=-1, keepdims=True))
      total_loss = loss + self.reg_factor*tf.reduce_mean(l2_norm)
    gradients = tape.gradient(total_loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
    metrics.update_running_metrics(y, preds)
    return loss

  @tf.function
  def test_step(self, x, y, metrics, training=False):
    """test step for a mini-batch"""
    preds = self.model(x, training=training)
    loss = self.loss(y, preds)
    metrics.update_running_metrics(y, preds)
    return loss
    

  def train_epoch(self, trainset, batch_size=128, shuffle=True, verbose=False, store=True):
    """train over all mini-batches and keep track of metrics"""

    # prepare data
    if shuffle:
      trainset.shuffle(buffer_size=batch_size)
    batch_dataset = trainset.batch(batch_size)
    num_batches = len(list(batch_dataset))

    # train loop over all mini-batches 
    start_time = time.time()
    running_loss = 0
    for i, (x, y) in enumerate(batch_dataset):      
      loss_batch = self.train_step(x, y, self.metrics['train'])
      self.metrics['train'].running_loss.append(loss_batch)
      running_loss += loss_batch
      progress_bar(i+1, num_batches, start_time, bar_length=30, loss=running_loss/(i+1))

    # store training metrics
    if store:
      if verbose:
        self.metrics['train'].update_print()
      else:
        self.metrics['train'].update()


  def evaluate(self, name, dataset, batch_size=128, verbose=True, training=False):
    """Evaluate model in mini-batches"""
    batch_dataset = dataset.batch(batch_size)
    num_batches = len(list(batch_dataset))
    for i, (x, y) in enumerate(batch_dataset):   
      loss_batch = self.test_step(x, y, self.metrics[name], training)
      self.metrics[name].running_loss.append(loss_batch)

    # store evaluation metrics
    if verbose:
      self.metrics[name].update_print()
    else:
      self.metrics[name].update()   
    

  def predict(self, x, batch_size=128):
    """Get predictions of model"""
    pred = self.model.predict(x, batch_size=batch_size)  
    return pred


  def set_early_stopping(self, patience=10, metric='loss', criterion=None):
    """set up early stopping"""
    self.early_stopping = EarlyStopping(patience=patience, metric=metric, criterion=criterion)
    

  def check_early_stopping(self, name='valid'):
    """check status of early stopping"""
    return self.early_stopping.status(self.metrics[name].get(self.early_stopping.metric)[-1])


  def set_lr_decay(self, decay_rate, patience, metric='loss', criterion=None):
    """set up learning rate decay"""
    self.lr_decay = LRDecay(optimizer=self.optimizer, decay_rate=decay_rate, 
                            patience=patience, metric=metric, criterion=criterion)

  def check_lr_decay(self, name='valid'):
    """check status and update learning rate decay"""
    self.lr_decay.check(self.metrics[name].get(self.lr_decay.metric)[-1])


  def get_metrics(self, name, metrics=None):
    """return a dictionary of metrics stored throughout training"""
    if metrics is None:
      metrics = {}
    metrics[name+'_loss'] = self.metrics[name].loss
    for metric_name in self.metrics[name].metric_names:
      metrics[name+'_'+metric_name] = self.metrics[name].get(metric_name)
    return metrics


  def set_learning_rate(self, learning_rate):
    """short-cut to set the learning rate"""
    self.optimizer.learning_rate.assign(learning_rate)



def fit_attr_prior(model, loss, optimizer, x_train, y_train, validation_data, verbose=True,  
                  metrics=['auroc', 'aupr'], num_epochs=100, batch_size=128, shuffle=True, 
                  reg_factor=1e-4, es_patience=10, es_metric='auroc', es_criterion='max',
                  lr_decay=0.3, lr_patience=3, lr_metric='auroc', lr_criterion='max'):


  # create tensorflow dataset
  trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  validset = tf.data.Dataset.from_tensor_slices(validation_data)

  # create trainer class
  trainer = CustomTrainer(model, loss, optimizer, metrics, reg_factor)
  trainer.set_lr_decay(decay_rate=lr_decay, patience=lr_patience, metric=lr_metric, criterion=lr_criterion)
  trainer.set_early_stopping(patience=es_patience, metric=es_metric, criterion=es_criterion)

  # train model
  for epoch in range(num_epochs):  
    sys.stdout.write("\rEpoch %d \n"%(epoch+1))
    
    # train over epoch
    trainer.train_epoch(trainset, batch_size=batch_size, shuffle=shuffle, verbose=False)

    # validation performance
    trainer.evaluate('valid', validset, batch_size=batch_size, verbose=verbose)

    # check learning rate decay
    trainer.check_lr_decay('valid')
   
    # check early stopping
    if trainer.check_early_stopping('valid'):
      print("Patience ran out... Early stopping.")
      break
  
  # compile history
  history = trainer.get_metrics('train')
  history = trainer.get_metrics('valid', history)

  return history, trainer
