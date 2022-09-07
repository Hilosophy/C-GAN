
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import skimage
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, Concatenate, Input, BatchNormalization, Embedding, Multiply, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2DTranspose, Conv2D
from keras.layers import Flatten, Dropout
from keras.datasets import mnist


""" CIFER10 """
IMG_SHAPE = (32,32,3)  # 入力のShape
NUM_CLASS = 10         # 分類クラスの数
SAVE_PATH = './CGAN_CIFER10_WGAN_GP'    # 保存フォルダのパス

LOAD_WEIGHTS = False
INIT_ITR = 8400


class CGAN():

    def __init__(self, save_path=SAVE_PATH, img_shape=IMG_SHAPE, num_class=NUM_CLASS, \
                img_grid=(10, NUM_CLASS), kernel_size=3, z_dim=500, d_label_channels=10, num_epoch=20, batch_size=32, lamd=10):

        # 画像の保存先
        self.path = save_path

        # 入力Imgサイズ
        self.img_shape = img_shape
        self.rows = img_shape[0]
        self.cols = img_shape[1]
        self.chans = img_shape[2]

        # 出力モンタージュ画像のShape
        self.img_grid = img_grid

        # 潜在変数の次元数
        self.z_dim  = z_dim

        # 勾配ペナルティの係数
        self.lamd = lamd

        # その他パラメータ
        self.d_label_channels = d_label_channels  # 識別器に入力する用の分類クラスをエンコードした画像に何チャンネル割り当てるか
        self.kernel_size = kernel_size
        self.num_class = num_class
        self.num_epoch = num_epoch
        self.batch_size = batch_size


    def build_G_layers(self):
        model = Sequential()
        # 第一層
        model.add(Dense(input_dim=self.z_dim+ self.num_class, units=1024, use_bias=False))  # output_shape=(1024,)

        # 第一層
        model.add(Dense(units=128* self.rows* self.cols /16, use_bias=False)) 
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.01))
        model.add(Reshape((int(self.rows/4), int(self.cols/4), 128)))  # output_shape=(8,8,128)

        # 第三層
        model.add(UpSampling2D(size=(2,2)))
        model.add(Conv2D(filters=64, kernel_size=self.kernel_size, strides=(1,1), padding='same', use_bias=False))  # output_shape=(16,16,64)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.01))   

        # 第四層
        model.add(UpSampling2D(size=(2,2)))
        model.add(Conv2D(filters=32, kernel_size=self.kernel_size, strides=(1,1), padding='same', use_bias=False))  # output_shape=(32,32,32)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.01))   


        # 第五層
        model.add(Conv2D(filters=self.chans, kernel_size=self.kernel_size, strides=(1,1), padding='same', use_bias=False, activation="tanh"))   # output_shape=(32,32,3)

        return model


    def build_G_model(self):
        """ 偽数字の生成用モデル """

        z = Input(shape=(self.z_dim,))            # 潜在ベクトル
        y_enc = Input(shape=(self.num_class,))    # one-hotにエンコードしたクラスラベル
        generator = self.build_G_layers()

        z_y = Concatenate(axis=1)([z, y_enc])
        gen_img = generator(z_y)  

        return Model(inputs=[z, y_enc], outputs=gen_img)




    def build_D_layers(self):
        model = Sequential()
        # 第一層
        model.add(Conv2D(filters=32, kernel_size=self.kernel_size, strides=(1,1), padding='same', input_shape=(self.rows, self.cols, self.d_label_channels+ self.chans), use_bias=False))  # BatchNormalizationの直前ではバイアスは冗長                                                                             
        model.add(LeakyReLU(0.2))                                                                               # output_shape=(32,32,32)

        # 第一層
        model.add(Conv2D(filters=64, kernel_size=self.kernel_size, strides=(2,2), padding='same', use_bias=False))       # output_shape=(16,16,64)
        model.add(BatchNormalization(momentum=0.8))                                                                                    
        model.add(LeakyReLU(0.2))

        # 第二層
        model.add(Conv2D(filters=128, kernel_size=self.kernel_size, strides=(2,2), padding='same', use_bias=False))  # output_shape=(8,8,128)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))

        # 第四層
        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        #model.add(Dropout(0.5))


        # 第五層
        model.add(Dense(1, activation=None))  # lossの計算の際にLogitを使いたいため, sigmoidは使わない

        return model


    def build_D_model(self):
        """ 識別器の訓練用モデル """       

        img = Input(shape=self.img_shape)                                          # 入力画像
        y_enc = Input(shape=(self.rows, self.cols, self.d_label_channels))         # クラスラベル 
        img_y = Concatenate(axis=3)([img, y_enc])                                  # input_shape=(32,32,self.d_label_channels+1)

        discriminator = self.build_D_layers()
        cls = discriminator(img_y)

        return Model(inputs=[img, y_enc], outputs=cls)


    def build_model(self):
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, beta_1=0.5, beta_2=0.9)
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5, beta_2=0.9)
        
        if tf.config.list_physical_devices('GPU'):
            device_name = tf.test.gpu_device_name()
        else:
            device_name = '/CPU:0'
        print(device_name)

        with tf.device(device_name):
            self.discriminator = self.build_D_model()  # discriminatorの訓練に使う
            self.generator = self.build_G_model()      # Generatorの訓練にはGenerator単体では使わない


    def encode_d(self, y):
        y_enc = np.zeros(self.rows* self.cols* self.d_label_channels)
        l = self.rows* self.cols* self.d_label_channels// self.num_class
        y_enc[int(l*y):int(l*(y+1))] = 1

        return y_enc.reshape((self.rows, self.cols, self.d_label_channels))


    def create_montage(self, imgs, name, grid_shape):

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        imgs = (imgs*127.5 + 127.5).astype(np.uint8)
        
        if imgs.shape[3] != 1:
            mont = skimage.util.montage(imgs, padding_width=1, fill=(255,255,255), channel_axis=3, grid_shape=grid_shape)
        else:
            mont = skimage.util.montage(imgs[:,:,:,0], padding_width=1, fill=255, grid_shape=grid_shape)

        skimage.io.imsave(os.path.join(self.path, name), mont)


    def plot_history(self, g_hist, d_hist):

        g_hist = np.array(g_hist)
        d_hist = np.array(d_hist)
  
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1,len(g_hist)+1), g_hist[:], label="Loss Generator", lw=3)
        ax.plot(np.arange(1,len(d_hist)+1), d_hist[:], label="Loss Discriminator", lw=3)
        ax.set_xlabel("Iteration", size=15)
        #ax.set_yscale("log")
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.path, 'train_history.png'))


    def read_data(self):
        (X_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
        X_train = (X_train.astype(np.float32) - 127.5)/127.5

        return X_train, y_train


    def calc_GP(self, r_img, f_img, y_enc_d):
        
        with tf.GradientTape() as GP_tape:
            alpha = tf.random.uniform(shape=[self.batch_size, 1, 1, 1], minval=0, maxval=1)
            rf_img = alpha* r_img+ (1- alpha)* f_img
            GP_tape.watch(rf_img)
            log_rf = self.discriminator([rf_img, y_enc_d])

        GP_grad = GP_tape.gradient(log_rf, [rf_img,])[0]
        GP_grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(GP_grad), axis=[1,2,3]))
        
        return tf.reduce_mean(tf.square(GP_grad_l2- 1))


    def train(self):

        X_train, y_train = self.read_data()

        g_hist = []
        d_hist = []
        imgs_hist = []

        for epoch in range(1, self.num_epoch+1):
            idx = np.random.randint(0, len(X_train), len(X_train))
            for i in range(int(X_train.shape[0] / self.batch_size)):
                itr = (epoch- 1)* int(X_train.shape[0] / self.batch_size)+ i

                if LOAD_WEIGHTS and i==0:
                    self.generator.load_weights(os.path.join(self.path, 'generator.h5')) 
                    self.discriminator.load_weights(os.path.join(self.path, 'discriminator.h5')) 
                
                if LOAD_WEIGHTS:
                    itr += INIT_ITR

                """ 生成画像を出力 """
                f_img_num = (self.img_grid[0]- 1)*self.img_grid[1]
                if itr % 300 == 0:
                    z_out = np.random.normal(-1, 1, (f_img_num, self.z_dim))    # 潜在ベクトル
                    y_out = np.arange(f_img_num)%10                             # 偽手書き数字用に、数字の分類クラスを0-9まで順番に並べ、9回繰り返す
                    y_out_enc = tf.keras.utils.to_categorical(y_out, self.num_class)                        # 生成器に入力する分類クラスyをone-hot表示

                    ex = np.array([X_train[(y_train==i).reshape(-1)][np.random.randint(0, len(X_train[(y_train==i).reshape(-1)]))] for i in range(self.num_class)])  # 本物の数字を0-9まで並べる
                    imgs = self.generator([z_out, y_out_enc])                                               # 生成器で偽の手書き数字生成
                    imgs = np.concatenate([ex, imgs], axis=0)                                               # 本物の数字が1行目に、偽者の数字が2-10行目に来るように並べる
                    self.create_montage(imgs, f"iter{itr}.png", self.img_grid)                              # モンタージュ画像の生成

                if itr % 1500 == 0 or itr in [300, 600, 900]:
                    imgs_hist.extend(imgs[10:20])


                """ 本物, 偽手書き数字の生成 """
                #z = np.random.uniform(-1, 1, (self.batch_size, self.z_dim))    # 潜在ベクトル（-1~1 均一分布）
                z = np.random.normal(0, 1, (self.batch_size, self.z_dim))       # 潜在ベクトル（μ0,σ1 正規分布）
                r_img = X_train[idx[i*self.batch_size:(i+1)*self.batch_size]]   # 実データの画像
                y = y_train[idx[i*self.batch_size:(i+1)*self.batch_size]]       # 実データの分類クラスy
                y_enc_g = tf.keras.utils.to_categorical(y, self.num_class)      # 生成器に入力する分類クラスyをone-hot表示
                y_enc_d = np.array(list(map(self.encode_d, y)))                 # 識別器に入力する分類クラスyを(28x28)画素に渡って0,1表示
                f_img = self.generator([z, y_enc_g], training=True)             # 生成器で偽の手書き数字生成

                """ 識別器の訓練 """
                with tf.GradientTape() as D_tape:
                    D_log_r = self.discriminator([r_img, y_enc_d], training=True)
                    D_loss_r = -tf.math.reduce_mean(D_log_r)
                    D_log_f = self.discriminator([f_img, y_enc_d], training=True)
                    D_loss_f = tf.math.reduce_mean(D_log_f)
                    
                    GP_loss = self.calc_GP(r_img, f_img, y_enc_d)

                    D_loss = D_loss_r + D_loss_f + GP_loss* self.lamd

                D_grads = D_tape.gradient(D_loss, self.discriminator.trainable_variables)     # discriminatorの訓練可能変数に対して微分して勾配求める
                self.D_optimizer.apply_gradients(grads_and_vars=zip(D_grads, self.discriminator.trainable_variables))
                
                """ 生成器の訓練 """    
                if itr% 5 == 0:        
                    z = np.random.normal(0, 1, (self.batch_size, self.z_dim))       # 潜在ベクトル（μ0,σ1 正規分布） 
                    with tf.GradientTape() as G_tape:
                        f_img = self.generator([z, y_enc_g], training=True)         # 生成器で偽の手書き数字生成
                        G_log_f = self.discriminator([f_img, y_enc_d], training=True)
                        G_loss = -tf.math.reduce_mean(G_log_f)

                    G_grads = G_tape.gradient(G_loss, self.generator.trainable_variables)  # generatorの訓練可能変数に対して微分して勾配求める
                    self.G_optimizer.apply_gradients(grads_and_vars=zip(G_grads, self.generator.trainable_variables))
                
                print(f"epoch: {epoch}, iteration: {itr}, g_loss: {G_loss:.4f}, d_loss: {D_loss:.4f}, d_loss_fake: {D_loss_f:.4f}, d_loss_real: {D_loss_r:.4f}, gp_loss: {GP_loss:.4f}")

                """ 損失の記録 """
                g_hist.append(G_loss)
                d_hist.append(D_loss)
                
            self.generator.save_weights(os.path.join(self.path, 'generator.h5'))                   # 各エポックごとに重みパラメータを保存更新
            self.discriminator.save_weights(os.path.join(self.path, 'discriminator.h5'))           # 各エポックごとに重みパラメータを保存更新
            self.create_montage(np.array(imgs_hist), "img_history.png", (len(imgs_hist)//10, 10))  # 指定のiteration時点での画像のモンタージュをプロットし、保存更新
            self.plot_history(g_hist, d_hist)                                                      # 各エポックごとにLossとAccの訓練推移をプロットし、保存更新

if __name__ == '__main__':
    cGAN = CGAN()
    cGAN.build_model()
    cGAN.train()