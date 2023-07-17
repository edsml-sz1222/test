import numpy as np
from matplotlib import pyplot as plt
import tensorflow.keras.layers
from tensorflow.keras.layers import Dense, Input, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from tensorflow.keras.losses import mse
from tensorflow.keras.models import Sequential
import time
import resource
import gc

from sklearn.utils import shuffle

from tensorflow.keras.backend import clear_session
from tensorflow.keras import layers
from tensorflow import compat
from tensorflow.python.keras import backend as K
import tensorflow as tf

#setings for limiting cores (if required)
# config = compat.v1.ConfigProto(intra_op_parallelism_threads=4,
#                         inter_op_parallelism_threads=4,
#                         allow_soft_placement=True)
#
# session = compat.v1.Session(config=config)
# K.set_session(session)

#default losses
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()
accuracy = tf.keras.metrics.BinaryAccuracy()
#samping for variational
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#custom losses
def full_models_loss(inputs, reconstruction, loss_weight):
    return loss_weight * mse(inputs, reconstruction)

def discriminator_loss(real_output, fake_output, loss_weight):
    loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return loss_weight * (loss_fake + loss_real)

def generator_loss(fake_output, loss_weight):
    return loss_weight * cross_entropy(tf.ones_like(fake_output), fake_output)

#surrogate_model class
class surrogate_model:
    #initialise with input/output size
    def __init__(self, input_size,output_size):
        self.input_size = input_size
        self.output_size = output_size

    #create architecture
    #nlayers = num of layers, real_layer can be 'Adversarial' or 'Variational', real_position is position of real layer,
    #def neurons can be used to determine the number of neurons in each layer (int or list), others are hyperparameters for layers
    def create_architecture(self,nlayers, real_layer = False, real_position = 0, def_neurons = False, beta = 0.001, drop_rate = 0.1,
                            loss_weights = [0.95,0.04,0.01],activation_function='elu', drop_out = None, use_bias = True,
                            regularizer = None, real_number =False):
        nlayers -= 1
        self.beta = beta
        self.loss_weights = loss_weights
        self.real_layer = real_layer
        self.type = real_layer
        #Input layer
        Input_img = Input(shape=(self.input_size,))
        if def_neurons == False:
            layer = Dense(self.input_size, activation=activation_function, name='layer0',kernel_regularizer=regularizer, use_bias=use_bias)(Input_img)
        else:
            try:
                layer = Dense(def_neurons[0], activation=activation_function, name='laye0',kernel_regularizer=regularizer, use_bias=use_bias)(Input_img)
            except:
                layer = Dense(def_neurons, activation=activation_function, name='layer0',kernel_regularizer=regularizer, use_bias=use_bias)(Input_img)
        i = 0
        while i < nlayers:
            i += 1
            if i == nlayers:
                layer = Dense(self.output_size, activation=activation_function, name='layer' + str(i),kernel_regularizer=regularizer, use_bias=use_bias)(layer)
            else:
                if drop_out != None:
                    layer = Dropout(drop_out)(layer)
                else:
                    None
                if (real_layer != False) and (real_position == i):
                    if real_layer == 'Adversarial':
                        # real adversarial layers
                        front_model = Model(Input_img,layer)
                        if real_number == False:
                            if def_neurons == False:
                                self.nlatent = self.input_size
                                Input_img_mid = Input(shape=(self.input_size,))
                                layer = Dense(self.input_size, activation=activation_function, name='layer' + str(i),kernel_regularizer=regularizer, use_bias=use_bias)(Input_img_mid)
                            else:
                                try:
                                    self.nlatent = def_neurons[i-1]
                                    Input_img_mid = Input(shape=(def_neurons[i-1],))
                                    layer = Dense(def_neurons[i], activation=activation_function, name='layer' + str(i),kernel_regularizer=regularizer, use_bias=use_bias)(Input_img_mid)
                                except:
                                    self.nlatent = def_neurons
                                    Input_img_mid = Input(shape=(def_neurons,))
                                    layer = Dense(def_neurons, activation=activation_function, name='layer' + str(i),kernel_regularizer=regularizer, use_bias=use_bias)(Input_img_mid)
                        else:
                            try:
                                self.nlatent = def_neurons[i-1]
                            except:
                                self.nlatent = def_neurons
                            Input_img_mid = Input(shape=(self.nlatent,))
                            layer = Dense(real_number, activation=activation_function, name='layer' + str(i),
                                          kernel_regularizer=regularizer, use_bias=use_bias)(Input_img_mid)

                    elif real_layer == 'Variational':
                        #real variational layers
                        if real_number == False:
                            if def_neurons == False:
                                z_mean = Dense(self.input_size, name='z_mean')(layer)
                                z_log_var = Dense(self.input_size, name='z_log_var')(layer)
                                z = Sampling()([z_mean, z_log_var])
                            else:
                                try:
                                    z_mean = Dense(def_neurons[i], name='z_mean')(layer)
                                    z_log_var = Dense(def_neurons[i], name='z_log_var')(layer)
                                    z = Sampling()([z_mean, z_log_var])
                                except:
                                    z_mean = Dense(def_neurons, name='z_mean')(layer)
                                    z_log_var = Dense(def_neurons, name='z_log_var')(layer)
                                    z = Sampling()([z_mean, z_log_var])
                        else:
                            z_mean = Dense(real_number, name='z_mean')(layer)
                            z_log_var = Dense(real_number, name='z_log_var')(layer)
                            z = Sampling()([z_mean, z_log_var])
                        front_model = Model(Input_img, [z_mean, z_log_var, z], name='encoder')
                        if real_number == False:
                            if def_neurons == False:
                                Input_img_mid = Input(shape=(self.input_size,))
                                layer = Dense(self.input_size, activation=activation_function, name='layer' + str(i),kernel_regularizer=regularizer, use_bias=use_bias)(Input_img_mid)
                            else:
                                try:
                                    Input_img_mid = Input(shape=(def_neurons[i-1],))
                                    layer = Dense(def_neurons[i], activation=activation_function, name='layer' + str(i),kernel_regularizer=regularizer, use_bias=use_bias)(Input_img_mid)
                                except:
                                    Input_img_mid = Input(shape=(def_neurons,))
                                    layer = Dense(def_neurons, activation=activation_function, name='layer' + str(i),kernel_regularizer=regularizer, use_bias=use_bias)(Input_img_mid)
                        else:
                            Input_img_mid = Input(shape=(real_number,))
                            layer = Dense(real_number, activation=activation_function, name='layer' + str(i),
                                          kernel_regularizer=regularizer, use_bias=use_bias)(Input_img_mid)
                    else:
                        print('Not an acceptable real layer type')
                else:
                    if def_neurons == False:
                        layer = Dense(self.input_size, activation = activation_function, name = 'layer'+str(i),kernel_regularizer=regularizer, use_bias=use_bias)(layer)
                    else:
                        try:
                            layer = Dense(def_neurons[i], activation=activation_function, name='layer' + str(i),kernel_regularizer=regularizer, use_bias=use_bias)(layer)
                        except:
                            layer = Dense(def_neurons, activation=activation_function, name='layer' + str(i),kernel_regularizer=regularizer, use_bias=use_bias)(layer)

        if real_layer == 'Adversarial':
            # constructs front, back, combined and critic models
            self.front_model = front_model
            back_model = Model(Input_img_mid,layer,name='decoder')
            self.back_model = back_model
            full_model = Sequential([self.front_model,self.back_model])
            self.full_model = full_model
            if def_neurons == False:
                self.latent = self.input_size
                Input_img_disc = Input(shape=(self.input_size))
                layer = Dense(self.input_size)(Input_img_disc)
            else:
                try:
                    self.latent = def_neurons[real_position]
                    Input_img_disc = Input(shape=(def_neurons[real_position]))
                    layer = Dense(def_neurons[real_position])(Input_img_disc)
                except:
                    self.latent = def_neurons
                    Input_img_disc = Input(shape=(def_neurons))
                    layer = Dense(def_neurons)(Input_img_disc)
            layer = BatchNormalization()(layer)
            layer = LeakyReLU()(layer)
            if def_neurons == False:
                self.latent = self.input_size
                Input_img_disc = Input(shape=(self.input_size))
                layer = Dense(self.input_size)(Input_img_disc)
            else:
                try:
                    self.latent = def_neurons[real_position-1]
                    Input_img_disc = Input(shape=(def_neurons[real_position-1]))
                    layer = Dense(def_neurons[real_position])(Input_img_disc)
                except:
                    self.latent = def_neurons
                    Input_img_disc = Input(shape=(def_neurons))
                    layer = Dense(def_neurons)(Input_img_disc)
            layer = BatchNormalization()(layer)
            layer = LeakyReLU()(layer)
            layer = Dropout(drop_rate)(layer)
            layer = Dense(1, activation='sigmoid')(layer)
            disc = Model(Input_img_disc,layer)
            self.disc = disc
            Adv_input = Input(shape=(self.input_size))
            adv_front = self.front_model(Adv_input)
            adv_back_output = self.back_model(adv_front)
            disc_output = self.disc(adv_front)
            self.combined = Model(Adv_input, [adv_back_output, disc_output])
        elif real_layer == 'Variational':
            #constructs front and back models
            self.front_model = front_model
            back_model = Model(Input_img_mid, layer)
            self.back_model = back_model
        else:
            self.full_model = Model(Input_img, layer)

    #custom training step for adversarial model
    @tf.function
    def train_step_adv(self,batch_x, batch_y=None, training = True):
        with tf.GradientTape() as ae_tape:
            encoder_output = self.front_model(batch_x, training=True)
            decoder_output = self.back_model(encoder_output, training=True)
            # Autoencoder loss
            ae_loss = full_models_loss(batch_y, decoder_output, self.loss_weights[0])
        if training == True:
            ae_grads = ae_tape.gradient(ae_loss,
                                        self.front_model.trainable_variables + self.back_model.trainable_variables)
            self.ae_optimizer.apply_gradients(
                zip(ae_grads, self.front_model.trainable_variables + self.back_model.trainable_variables))

        # Discriminator
        with tf.GradientTape() as dc_tape:
            real_distribution = tf.random.normal([batch_x.shape[0], self.nlatent], mean=0.0, stddev=1.0)
            encoder_output = self.front_model(batch_x, training=True)

            dc_real = self.disc(real_distribution, training=True)
            dc_fake = self.disc(encoder_output, training=True)

            # Discriminator Loss
            dc_loss = discriminator_loss(dc_real, dc_fake, self.loss_weights[2])

            # Discriminator Acc
            dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                              tf.concat([dc_real, dc_fake], axis=0))
        if training == True:
            dc_grads = dc_tape.gradient(dc_loss, self.disc.trainable_variables)
            self.dc_optimizer.apply_gradients(zip(dc_grads, self.disc.trainable_variables))

        # Generator (Encoder)
        with tf.GradientTape() as gen_tape:
            encoder_output = self.front_model(batch_x, training=True)
            dc_fake = self.disc(encoder_output, training=True)

            # Generator loss
            gen_loss = generator_loss(dc_fake, self.loss_weights[1])
        if training == True:
            gen_grads = gen_tape.gradient(gen_loss, self.front_model.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gen_grads, self.front_model.trainable_variables))

        return ae_loss, dc_loss, dc_acc, gen_loss

    # custom training step for variational model
    @tf.function
    def train_step_var(self, batch_x, batch_y=None, training=True):
        # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        with tf.GradientTape() as ae_tape:
            z_mean, z_log_var, z = self.front_model(batch_x,training=training)
            reconstruction = self.back_model(z, training=training)
            reconstruction = K.flatten(reconstruction)
            y_flatten = K.flatten(batch_y)
            reconstruction_loss = mse(y_flatten, reconstruction)
            kl_loss = -5e-4 * self.beta*K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)
        if training == True:
            ae_grads = ae_tape.gradient(total_loss,
                                        self.front_model.trainable_variables + self.back_model.trainable_variables)
            self.ae_optimizer.apply_gradients(
                zip(ae_grads, self.front_model.trainable_variables + self.back_model.trainable_variables))

        return reconstruction_loss, kl_loss, total_loss
    # custom training step for normal model
    @tf.function
    def train_step_norm(self, batch_x, batch_y=None, training=True):

        with tf.GradientTape() as ae_tape:
            output = self.full_model(batch_x)
            reconstruction = K.flatten(output)
            y_flatten = K.flatten(batch_y)
            reconstruction_loss = mse(y_flatten, reconstruction)
        if training == True:
            ae_grads = ae_tape.gradient(reconstruction_loss,
                                        self.full_model.trainable_variables)
            self.ae_optimizer.apply_gradients(
                zip(ae_grads, self.full_model.trainable_variables))

        return reconstruction_loss

    #single epoch of back propogation
    def pred_back_step(self,input_guess,real_output,specify_input=None,start_loss = 0,end_loss = -1, mini_batch = False
                       , constraint = False, con_wei = 0, ncon_num=5):
        #input guess is the single set of model coeffecients, real_output is the known solution for multiple experiments,
        #specify input is an optional input that specifies a part of the input, start and end loss determine the part of the
        #output where mse is applied
        #mini batch determines when the loss is updated
        mid_point = np.zeros((1,input_guess.shape[1]-ncon_num))
        mid_point[:,:]=1/2.0
        # mid_point = tf.Variable(mid_point)
        if mini_batch == False:
            with tf.GradientTape() as tape:
                #assign tape to watch the input_guess
                tape.watch(input_guess)
                #iterate across all experiments recieved
                for i in range(real_output.shape[0]):
                    #assign the first 5 variables in the input_guess to be the experimental conditions
                    if specify_input.any() ==None:
                        None
                    else:
                        for j10 in range(ncon_num):
                            input_guess[0,j10].assign(specify_input[i,j10])
                    #predict the output solution from the 5 experimental conditions + 27 model coeffecients
                    network_output = self.full_model(input_guess, training=False)
                    #determine the MSE between the known solution for a given experiment and the one produced above
                    if i == 0:
                        loss = mse(real_output[i,int(start_loss):int(end_loss)],network_output[0,int(start_loss):int(end_loss)])
                    else:
                        loss = loss + mse(real_output[i,int(start_loss):int(end_loss)], network_output[0,int(start_loss):int(end_loss)])

                #determine the mean loss across all experiments
                loss = loss/real_output.shape[0]
                if constraint == True:
                    if specify_input.any()==None:
                        con_loss = mse(input_guess[0, :], mid_point[0, :])
                    else:
                        con_loss = mse(input_guess[0,ncon_num:],mid_point[0,:])
                    con_loss = tf.cast(con_loss, tf.float32)
                    loss = loss + con_wei*con_loss
            #determine the gradient of the loss applied to the input guess
            gradient = tape.gradient(loss,input_guess)
            #change the input guess based on this gradient
            self.ae_optimizer.apply_gradients(zip([gradient],[input_guess]))
            #if any model coeffecient deviates from range, adjust it back (ignore parameters)
            for i in range(ncon_num,input_guess.shape[1]):
                if input_guess[0, i] > 1:
                    input_guess[0, i].assign(0.999)
                elif input_guess[0, i] < 0:
                    input_guess[0, i].assign(0.001)
                else:
                    None
        #same as above but input guess is modified after each experiment rather than after all experiments
        elif mini_batch == True:
            for i in range(real_output.shape[0]):
                with tf.GradientTape() as tape:
                    tape.watch(input_guess)
                    if specify_input.any() ==None:
                        None
                    else:
                        for j10 in range(ncon_num):
                            input_guess[0, j10].assign(specify_input[i, j10])

                    network_output = self.full_model(input_guess, training=False)
                    loss = mse(real_output[i, int(start_loss):int(end_loss)],
                                   network_output[0, int(start_loss):int(end_loss)])

                gradient = tape.gradient(loss, input_guess)
                self.ae_optimizer.apply_gradients(zip([gradient], [input_guess]))
        return loss
    #predict input
    def predict_input(self,real_output,epoch = 2000,specify_input=None,start_loss =0,end_loss = -1,mini_batch=False,constraint=True,
                      con_wei=0, given_coeffs = False):
        #real_output is the known output
        #create optimizer
        self.ae_optimizer = tf.keras.optimizers.Adam()
        #create random input space
        try:
            if given_coeffs==False:
                initial_guess = tf.random.normal([1, self.input_size], mean=0.0, stddev=0.3)
            else:
                initial_guess =  tf.convert_to_tensor(given_coeffs)
        except:
            initial_guess = tf.convert_to_tensor(given_coeffs)
            #turn input space into a trainable variables
        initial_guess = tf.Variable(initial_guess)
        loss = []
        #iterate through pred_back_step for the number of epochs, updating loss. Returns the input space
        for i in range(epoch):
            loss.append(self.pred_back_step(initial_guess,real_output,specify_input = specify_input,start_loss=start_loss,
                                            end_loss=end_loss,mini_batch=mini_batch,constraint=constraint,con_wei=con_wei).numpy())
        self.prediction_loss = loss
        return initial_guess


    #training model algorithm
    def train_model(self,epochs,training_x,training_y,test_x=None,test_y=None, optimizer = 'Nadam', callbacks = None, loss = 'mean_squared_error', 
                    batch_size = 200, save_full_model = False, save_location = 'model', shuffle_data=True,
                    custom_training=True, early_stopping = False, LRonPlat=False):
        if self.type == 'Variational':
            if custom_training==True:
                base_lr = 0.00025
                if optimizer == 'Adam':
                    self.ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
                elif optimizer == 'Nadam':
                    self.ae_optimizer = tf.keras.optimizers.Nadam(lr=base_lr)
                # tf.config.experimental_run_functions_eagerly(True)
                # Training loop
                n_epochs = epochs
                losses = np.zeros((1, 6))
                max_lr = 0.00025
                n_samples = training_x.shape[0]
                step_size = 2 * np.ceil(n_samples / batch_size)
                global_step = 0
                for epoch in range(n_epochs):
                    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    start = time.time()
                    train_epoch_ae_loss_avg = tf.metrics.Mean()
                    train_epoch_dc_loss_avg = tf.metrics.Mean()
                    train_epoch_dc_acc_avg = tf.metrics.Mean()
                    test_epoch_ae_loss_avg = tf.metrics.Mean()
                    test_epoch_dc_loss_avg = tf.metrics.Mean()
                    test_epoch_dc_acc_avg = tf.metrics.Mean()
                    batch_num = int(training_x.shape[0] / batch_size)
                    if shuffle_data == True:
                        training_x,training_y = shuffle(training_x,training_y)
                    for j1 in range(batch_num):
                        # -------------------------------------------------------------------------------------------------------------
                        train_batch_x = tf.convert_to_tensor(training_x[batch_size * j1:batch_size * (j1 + 1), :])
                        train_batch_y = tf.convert_to_tensor(training_y[batch_size * j1:batch_size * (j1 + 1), :])
                        global_step = global_step + 1
                        cycle = np.floor(1 + global_step / (2 * step_size))
                        x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                        clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr)
                        try:
                            self.ae_optimizer.lr = clr

                        except:
                            None
                        train_ae_loss, train_dc_loss, train_dc_acc = self.train_step_var(train_batch_x, batch_y=train_batch_y)

                        train_epoch_ae_loss_avg(train_ae_loss)
                        train_epoch_dc_loss_avg(train_dc_loss)
                        train_epoch_dc_acc_avg(train_dc_acc)
                    if shuffle_data == True:
                        test_x,test_y = shuffle(test_x,test_y)
                    batch_num = int(test_x.shape[0] / batch_size)
                    for j1 in range(batch_num):
                        test_batch_x = tf.convert_to_tensor(test_x[batch_size * j1:batch_size * (j1 + 1), :])
                        test_batch_y = tf.convert_to_tensor(test_y[batch_size * j1:batch_size * (j1 + 1), :])
                        test_ae_loss, test_dc_loss, test_dc_acc = self.train_step_var(test_batch_x, batch_y=test_batch_y,
                                                                                     training=False)
                        test_epoch_ae_loss_avg(test_ae_loss)
                        test_epoch_dc_loss_avg(test_dc_loss)
                        test_epoch_dc_acc_avg(test_dc_acc)
                    epoch_time = time.time() - start
                    print('done')
                    print('{:4d}: TIME: {:.2f} ETA: {:.2f} TRAIN_LOSS: {:.4f} TEST_LOSS: {:.4f}' \
                          .format(epoch, epoch_time,
                                  epoch_time * (n_epochs - epoch),
                                  train_epoch_dc_acc_avg.result(),
                                  test_epoch_dc_acc_avg.result()))
                    if epoch == 0:
                        losses[epoch, 0] = train_epoch_ae_loss_avg.result()
                        losses[epoch, 1] = train_epoch_dc_loss_avg.result()
                        losses[epoch, 2] = train_epoch_dc_acc_avg.result()
                        losses[epoch, 3] = test_epoch_ae_loss_avg.result()
                        losses[epoch, 4] = test_epoch_dc_loss_avg.result()
                        losses[epoch, 5] = test_epoch_dc_acc_avg.result()
                    else:
                        temp_loss = np.zeros((1, 6))
                        temp_loss[0, 0] = train_epoch_ae_loss_avg.result()
                        temp_loss[0, 1] = train_epoch_dc_loss_avg.result()
                        temp_loss[0, 2] = train_epoch_dc_acc_avg.result()
                        temp_loss[0, 3] = test_epoch_ae_loss_avg.result()
                        temp_loss[0, 4] = test_epoch_dc_loss_avg.result()
                        temp_loss[0, 5] = test_epoch_dc_acc_avg.result()
                        losses = np.concatenate((losses, temp_loss))

                    # self.ae_optimizer.lr.assign(base_lr)
                    if epoch>60 and LRonPlat==True:
                        if (losses[epoch,3]-np.mean(losses[epoch-50:epoch,3]))<0:
                            base_lr = base_lr*0.2
                            self.ae_optimizer.lr.assign(base_lr)
                    if epoch>1000 and early_stopping==True:
                        if (losses[epoch,3]-np.mean(losses[epoch-50:epoch,3]))<0:
                            break
                    gc.collect()
                    clear_session()
                    compat.v1.reset_default_graph()
                self.losses = losses
            else:
                self.full_model = VSM(self.front_model,self.back_model)
                print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                print('compiling')
                self.full_model.compile(optimizer =optimizer)
                print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                print('compiled')
                self.history = self.full_model.fit(training_x,training_y, epochs=epochs, batch_size=training_x.shape[0], shuffle=True,
                                                    validation_data=(test_x, test_y), verbose=2, callbacks=callbacks)
        elif self.type == 'Adversarial':
            base_lr = 0.00025
            if optimizer == 'Adam':
                self.ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
            elif optimizer == 'Nadam':
                self.ae_optimizer = tf.keras.optimizers.Nadam(lr=base_lr)
            if optimizer == 'Adam':
                self.dc_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
            elif optimizer == 'Nadam':
                self.dc_optimizer = tf.keras.optimizers.Nadam(lr=base_lr)
            if optimizer == 'Adam':
                self.gen_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
            elif optimizer == 'Nadam':
                self.gen_optimizer = tf.keras.optimizers.Nadam(lr=base_lr)
            # Training loop
            n_epochs = epochs
            losses = np.zeros((1,8))
            base_lr = 0.00025
            max_lr = 0.00025
            n_samples = training_x.shape[0]
            step_size = 2 * np.ceil(n_samples / batch_size)
            global_step = 0
            if custom_training == True:
                for epoch in range(n_epochs):
                    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    start = time.time()
                    train_epoch_ae_loss_avg = tf.metrics.Mean()
                    train_epoch_dc_loss_avg = tf.metrics.Mean()
                    train_epoch_dc_acc_avg = tf.metrics.Mean()
                    train_epoch_gen_loss_avg = tf.metrics.Mean()
                    test_epoch_ae_loss_avg = tf.metrics.Mean()
                    test_epoch_dc_loss_avg = tf.metrics.Mean()
                    test_epoch_dc_acc_avg = tf.metrics.Mean()
                    test_epoch_gen_loss_avg = tf.metrics.Mean()
                    batch_num = int(training_x.shape[0] / batch_size)
                    if shuffle_data == True:
                        training_x,training_y = shuffle(training_x,training_y)
                    for j1 in range(batch_num):
                        # -------------------------------------------------------------------------------------------------------------
                        batch_x = training_x[batch_size * j1:batch_size * (j1 + 1), :]
                        batch_y = training_y[batch_size * j1:batch_size * (j1 + 1), :]
                        global_step = global_step + 1
                        cycle = np.floor(1 + global_step / (2 * step_size))
                        x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                        clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr)
                        try:
                            self.ae_optimizer.lr = clr
                            self.dc_optimizer.lr = clr
                            self.gen_optimizer.lr = clr
                        except:
                            None
                        ae_loss, dc_loss, dc_acc, gen_loss = self.train_step_adv(batch_x, batch_y = batch_y)

                        train_epoch_ae_loss_avg(ae_loss)
                        train_epoch_dc_loss_avg(dc_loss)
                        train_epoch_dc_acc_avg(dc_acc)
                        train_epoch_gen_loss_avg(gen_loss)
                    batch_num = int(test_x.shape[0] / batch_size)
                    if shuffle_data == True:
                        test_x,test_y = shuffle(test_x,test_y)
                    for j1 in range(batch_num):
                        batch_x = test_x[batch_size * j1:batch_size * (j1 + 1), :]
                        batch_y = test_y[batch_size * j1:batch_size * (j1 + 1), :]
                        ae_loss, dc_loss, dc_acc, gen_loss = self.train_step_adv(batch_x, batch_y=batch_y, training=False)

                        test_epoch_ae_loss_avg(ae_loss)
                        test_epoch_dc_loss_avg(dc_loss)
                        test_epoch_dc_acc_avg(dc_acc)
                        test_epoch_gen_loss_avg(gen_loss)
                    epoch_time = time.time() - start
                    print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_TRAIN_LOSS: {:.4f} DC_TRAIN_LOSS: {:.4f} GEN_TRAIN_LOSS: {:.4f} AE_TEST_LOSS: {:.4f} DC_TEST_LOSS: {:.4f} GEN_TEST_LOSS: {:.4f}' \
                          .format(epoch, epoch_time,
                                  epoch_time * (n_epochs - epoch),
                                  train_epoch_ae_loss_avg.result(),
                                  train_epoch_dc_loss_avg.result(),
                                  train_epoch_gen_loss_avg.result(),
                                  test_epoch_ae_loss_avg.result(),
                                  test_epoch_dc_loss_avg.result(),
                                  test_epoch_gen_loss_avg.result()))
                    if epoch == 0:
                        losses[epoch, 0] = train_epoch_ae_loss_avg.result()
                        losses[epoch, 1] = train_epoch_dc_loss_avg.result()
                        losses[epoch, 2] = train_epoch_dc_acc_avg.result()
                        losses[epoch, 3] = train_epoch_gen_loss_avg.result()
                        losses[epoch, 4] = test_epoch_ae_loss_avg.result()
                        losses[epoch, 5] = test_epoch_dc_loss_avg.result()
                        losses[epoch, 6] = test_epoch_dc_acc_avg.result()
                        losses[epoch, 7] = test_epoch_gen_loss_avg.result()
                    else:
                        temp_loss = np.zeros((1,8))
                        temp_loss[0, 0] = train_epoch_ae_loss_avg.result()
                        temp_loss[0, 1] = train_epoch_dc_loss_avg.result()
                        temp_loss[0, 2] = train_epoch_dc_acc_avg.result()
                        temp_loss[0, 3] = train_epoch_gen_loss_avg.result()
                        temp_loss[0, 4] = test_epoch_ae_loss_avg.result()
                        temp_loss[0, 5] = test_epoch_dc_loss_avg.result()
                        temp_loss[0, 6] = test_epoch_dc_acc_avg.result()
                        temp_loss[0, 7] = test_epoch_gen_loss_avg.result()
                        losses = np.concatenate((losses,temp_loss))
                    if epoch > 20 and LRonPlat==True:
                        if (losses[epoch, 4] - np.mean(losses[epoch - 20:epoch, 4])) < 0:
                            base_lr = base_lr * 0.2
                            self.ae_optimizer.lr.assign(base_lr)
                            self.gen_optimizer.lr.assign(base_lr)
                            self.dc_optimizer.lr.assign(base_lr)
                    if epoch > 1000 and early_stopping == True:
                        if (losses[epoch, 4] - np.mean(losses[epoch - 50:epoch, 4])) < 0:
                            break
                    gc.collect()
                    clear_session()
                    compat.v1.reset_default_graph()
                self.losses = losses
            else:
                losses = np.zeros((1, 6))
                self.disc.compile(loss='binary_crossentropy',optimizer = self.dc_optimizer)
                self.disc.trainable=False
                Adv_input = Input(shape=(self.input_size))
                encoded_output=self.front_model(Adv_input)
                decoder_output=self.back_model(encoded_output)
                valdity = self.disc(encoded_output)
                adv_net = Model(Adv_input,[decoder_output,valdity],name='Adv_net')
                adv_net.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.01], optimizer=self.ae_optimizer)
                for epoch in range(n_epochs):
                    start = time.time()

                    batch_num = int(training_x.shape[0] / batch_size)
                    if shuffle_data == True:
                        training_x, training_y = shuffle(training_x, training_y)
                    for j1 in range(batch_num):
                        batch_x = training_x[batch_size * j1:batch_size * (j1 + 1), :]
                        batch_y = training_y[batch_size * j1:batch_size * (j1 + 1), :]
                        real_distribution = tf.random.normal([batch_x.shape[0], self.nlatent], mean=0.0, stddev=1.0)
                        real = np.ones(batch_x.shape[0])
                        fake = np.zeros(batch_x.shape[0])
                        latent_fake = self.front_model.predict(batch_x)
                        g_train_loss = adv_net.train_on_batch(batch_x, [batch_y, real])
                        print('done')
                        print(str(g_train_loss))
                    batch_num = int(test_x.shape[0] / batch_size)
                    if shuffle_data == True:
                        test_x, test_y = shuffle(test_x, test_y)
                    for j1 in range(batch_num):
                        batch_x = test_x[batch_size * j1:batch_size * (j1 + 1), :]
                        batch_y = test_y[batch_size * j1:batch_size * (j1 + 1), :]
                        g_test_loss = adv_net.test_on_batch(batch_x, [batch_y, real])
                    if epoch == 0:
                        losses[epoch,:3] = g_train_loss[:]
                        losses[epoch, 3:] = g_test_loss[:]

                    else:
                        temp_loss = np.zeros((1, 6))
                        temp_loss[0,:3] = g_train_loss[:]
                        temp_loss[0, 3:] = g_test_loss[:]
                        losses = np.concatenate((losses, temp_loss))
                    gc.collect()
                    clear_session()
                    compat.v1.reset_default_graph()
            self.losses=losses


        else:
            if custom_training==True:
                base_lr = 0.00025
                if optimizer == 'Adam':
                    self.ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
                elif optimizer == 'Nadam':
                    self.ae_optimizer = tf.keras.optimizers.Nadam(lr=base_lr)
                # Training loop
                n_epochs = epochs
                losses = np.zeros((1, 2))
                max_lr = 0.00025
                n_samples = training_x.shape[0]
                step_size = 2 * np.ceil(n_samples / batch_size)
                global_step = 0
                for epoch in range(n_epochs):
                    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    start = time.time()
                    train_epoch_ae_loss_avg = tf.metrics.Mean()
                    test_epoch_ae_loss_avg = tf.metrics.Mean()
                    batch_num = int(training_x.shape[0] / batch_size)
                    if shuffle_data == True:
                        training_x,training_y = shuffle(training_x,training_y)
                    for j1 in range(batch_num):
                        # -------------------------------------------------------------------------------------------------------------
                        batch_x = training_x[batch_size * j1:batch_size * (j1 + 1), :]
                        batch_y = training_y[batch_size * j1:batch_size * (j1 + 1), :]
                        global_step = global_step + 1
                        cycle = np.floor(1 + global_step / (2 * step_size))
                        x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                        clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr)
                        try:
                            self.ae_optimizer.lr = clr

                        except:
                            None
                        ae_loss = self.train_step_norm(batch_x, batch_y=batch_y)

                        train_epoch_ae_loss_avg(ae_loss)
                    batch_num = int(test_x.shape[0] / batch_size)
                    if shuffle_data == True:
                        test_x,test_y = shuffle(test_x,test_y)
                    for j1 in range(batch_num):
                        # -------------------------------------------------------------------------------------------------------------
                        batch_x = test_x[batch_size * j1:batch_size * (j1 + 1), :]
                        batch_y = test_y[batch_size * j1:batch_size * (j1 + 1), :]
                        ae_loss = self.train_step_norm(batch_x, batch_y=batch_y, training=False)

                        test_epoch_ae_loss_avg(ae_loss)
                    epoch_time = time.time() - start
                    print('{:4d}: TIME: {:.2f} ETA: {:.2f} MSE_TRAIN_LOSS: {:.4f} MSE_TEST_LOSS: {:.4f}' \
                          .format(epoch, epoch_time,
                                  epoch_time * (n_epochs - epoch),
                                  train_epoch_ae_loss_avg.result(),
                                  test_epoch_ae_loss_avg.result()))
                    if epoch == 0:
                        losses[epoch, 0] = train_epoch_ae_loss_avg.result()
                        losses[epoch, 1] = test_epoch_ae_loss_avg.result()
                    else:
                        temp_loss = np.zeros((1, 2))
                        temp_loss[0, 0] = train_epoch_ae_loss_avg.result()
                        temp_loss[0, 1] = test_epoch_ae_loss_avg.result()
                        losses = np.concatenate((losses, temp_loss))
                    if epoch > 60 and LRonPlat==True:
                        if (losses[epoch, 0] - np.mean(losses[epoch - 50:epoch, 0])) > 0:
                            base_lr = base_lr * 0.2
                            self.ae_optimizer.lr.assign(base_lr)
                    if epoch > 500 and early_stopping == True:
                        if (losses[epoch, 1] - np.mean(losses[epoch - 50:epoch, 1])) > 0:
                            break
                    gc.collect()
                    clear_session()
                    compat.v1.reset_default_graph()
                self.losses = losses
            else:
                self.full_model.compile(optimizer, loss)
                self.history = self.full_model.fit(training_x,training_y, epochs=epochs, batch_size=batch_size, shuffle=True,
                                                    validation_data=(test_x, test_y), verbose=2, callbacks=callbacks)

        try:
            plt.yscale('log')
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.savefig(save_location + 'losses.png')
        except:
            try:
                plt.yscale('log')
                if self.type=='Adversarial':
                    plt.plot(self.losses[:, 0])
                    plt.plot(self.losses[:, 4])
                elif self.type=='Variational':
                    plt.plot(self.losses[:, 0])
                    plt.plot(self.losses[:, 3])
                else:
                    plt.plot(self.losses[:, 0])
                    plt.plot(self.losses[:, 1])
                plt.savefig(save_location + 'losses.png')
            except:
                print('could not plot losses')

        if save_full_model == True:
            try:
                self.full_model.save(save_location + 'full.h5')
            except:
                print('could not save full model')
            if (self.real_layer == 'Adversarial') or (self.real_layer == 'Variational'):
                self.front_model.save(save_location + 'front.h5')
                self.back_model.save(save_location + 'back.h5')
        else:
            if (self.real_layer == 'Adversarial') or (self.real_layer == 'Variational'):
                self.front_model.save_weights(save_location + 'front.h5')
                self.back_model.save_weights(save_location + 'back.h5')
            try:
                self.full_model.save_weights(save_location + 'full.h5')
            except:
                print('could not save full model')
    #loads previously trained model, if load_full_model = False loads just the weights (needs to have called create_architecture)
    def load_model(self,load_full_model=True,save_location='model'):
        if load_full_model == True:
            try:
                if self.real_layer == 'Adversarial':
                    self.front_model =tensorflow.keras.models.load_model(save_location + 'front.h5')
                    self.back_model =tensorflow.keras.models.load_model(save_location + 'back.h5')
                    Adv_input = Input(shape=(self.input_size))
                    encoder_ouputs = self.front_model(Adv_input)
                    decoder_ouputs = self.back_model(encoder_ouputs)
                    self.full_model = Model(Adv_input, decoder_ouputs)
                elif self.real_layer == 'Variational':
                    self.full_model =tensorflow.keras.models.load_model(save_location + '.h5')
            except:
                self.full_model = tensorflow.keras.models.load_model(save_location + 'full.h5')
        elif load_full_model == False:
            if self.real_layer == 'Adversarial':
                # self.full_model.load_weights(save_location + '.h5')
                self.front_model.load_weights(save_location + 'front.h5', by_name = True)
                self.back_model.load_weights(save_location + 'back.h5', by_name = True)
                Adv_input = Input(shape=(self.input_size))
                encoder_ouputs = self.front_model(Adv_input)
                decoder_ouputs = self.back_model(encoder_ouputs)
                self.full_model = Model(Adv_input, decoder_ouputs)
            elif self.real_layer == 'Variational':
                self.front_model.load_weights(save_location + 'front.h5')
                self.back_model.load_weights(save_location + 'back.h5')
                Adv_input = Input(shape=(self.input_size))
                encoder_ouputs = self.front_model(Adv_input)[2]
                decoder_ouputs = self.back_model(encoder_ouputs)
                self.full_model = Model(Adv_input, decoder_ouputs)
            else:
                self.full_model.load_weights(save_location + 'full.h5')
            # self.full_model.load_weights(save_location + '.h5')

#Alternative Variational class for training
class VSM(tensorflow.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VSM, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker =tensorflow.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker =tensorflow.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker =tensorflow.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    def call(self, x):
        None
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction = K.flatten(reconstruction)
            y_flatten = K.flatten(y)
            reconstruction_loss = mse(y_flatten, reconstruction)
            kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        gc.collect()
        # clear_session()
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x, training=False)
            reconstruction = self.decoder(z, training=False)
            reconstruction = K.flatten(reconstruction)
            y_flatten = K.flatten(y)
            reconstruction_loss = mse(y_flatten, reconstruction)
            kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)
        # grads = tape.gradient(total_loss, self.trainable_weights)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

