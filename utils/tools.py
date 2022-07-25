# Copyright 2021 Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
# import torch.nn as nn

def learn_scheduler(lr_dec, lr):
    def learning_scheduler_fn(epoch):
        lr_new = lr * (lr_dec ** epoch)
        return lr_new if lr_new >= 5e-5 else 5e-5
    return learning_scheduler_fn


def get_callbacks(tb_log_save_path, saved_model_path, lr_dec, lr):
    tb = tf.keras.callbacks.TensorBoard(log_dir=tb_log_save_path, histogram_freq=0)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(saved_model_path, monitor='val_Efficient_CapsNet_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    lr_decay = tf.keras.callbacks.LearningRateScheduler(learn_scheduler(lr_dec, lr))

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_CapsNet_accuracy', factor=0.9,
                              patience=4, min_lr=0.00001, min_delta=0.0001, mode='max')

    return [tb, model_checkpoint, lr_decay]


# def marginLoss(y_true, y_pred):
#     lbd = 0.5
#     m_plus = 0.9
#     m_minus = 0.1
    
#     L = y_true * tf.square(tf.maximum(0., m_plus - y_pred)) + \
#     lbd * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))

#     return tf.reduce_mean(tf.reduce_sum(L, axis=1))

# def marginLoss(y_pred, y_true):
#     """
#     :param projections: torch.Tensor, shape [batch_size, projection_dim]
#     :param targets: torch.Tensor, shape [batch_size]
#     :return: torch.Tensor, scalar
#     """
#     temperature=0.07
#     device = torch.device("cuda") if y_pred.is_cuda else torch.device("cpu")

#     dot_product_tempered = torch.mm(y_pred, y_pred.T) / temperature
#     # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
#     exp_dot_tempered = (
#         torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
#     )

#     mask_similar_class = (y_true.unsqueeze(1).repeat(1, y_true.shape[0]) == targets).to(device)
#     mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
#     mask_combined = mask_similar_class * mask_anchor_out
#     cardinality_per_samples = torch.sum(mask_combined, dim=1)

#     log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
#     supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
    

#     return torch.mean(supervised_contrastive_loss_per_sample)

def marginLoss(y_pred,y_true,margin = 0.7):
    
    return tfa.losses.contrastive_loss(y_true,y_pred,margin)


def multiAccuracy(y_true, y_pred):
    label_pred = tf.argsort(y_pred,axis=-1)[:,-2:]
    label_true = tf.argsort(y_true,axis=-1)[:,-2:]
    
    acc = tf.reduce_sum(tf.cast(label_pred[:,:1]==label_true,tf.int8),axis=-1) + \
          tf.reduce_sum(tf.cast(label_pred[:,1:]==label_true,tf.int8),axis=-1)
    acc /= 2
    return tf.reduce_mean(acc,axis=-1)
