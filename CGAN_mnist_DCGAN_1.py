

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


""" MNIST """
IMG_SHAPE = (28,28,1)  # 入力のShape
NUM_CLASS = 10         # 分類クラスの数
SAVE_PATH = './CGAN_mnist_DCGAN_1'    # 保存フォルダのパス


class CGAN():

    def __init__(self, save_path=SAVE_PATH, img_shape=IMG_SHAPE, num_class=NUM_CLASS, \
                img_grid=(10, NUM_CLASS), kernel_size=5, z_dim=100, d_label_channels=1, num_epoch=7, batch_size=32):

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

        # その他パラメータ
        self.d_label_channels = d_label_channels  # 識別器に入力する用の分類クラスをエンコードした画像に何チャンネル割り当てるか
        self.kernel_size = kernel_size
        self.num_class = num_class
        self.num_epoch = num_epoch
        self.batch_size = batch_size


    def build_G_layers(self):
        model = Sequential()
        # 第一層
        model.add(Dense(input_dim=self.z_dim+ self.num_class, units=128* self.rows* self.cols /16, use_bias=False)) 
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.01))
        model.add(Reshape((int(self.rows/4), int(self.cols/4), 128)))                      # output_shape=(7,7,128)

        # 第三層
        model.add(Conv2DTranspose(filters=64, kernel_size=self.kernel_size, strides=(2,2), padding='same', use_bias=False))  # output_shape=(14,14,64)
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.01))   

        # 第四層
        model.add(Conv2DTranspose(filters=self.chans, kernel_size=self.kernel_size, strides=(2,2), padding='same', use_bias=False, activation="tanh"))   # output_shape=(28,28,1)

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
        model.add(Conv2D(filters=64, kernel_size=self.kernel_size, strides=(2,2), padding='same', \
                        input_shape=(self.rows, self.cols, self.d_label_channels+ self.chans), use_bias=False))  # output_shape=(14,14,64)
        model.add(BatchNormalization())                                                                                    
        model.add(LeakyReLU(0.2))

        # 第二層
        model.add(Conv2D(filters=128, kernel_size=self.kernel_size, strides=(2,2), padding='same', use_bias=False))  # output_shape=(7,7,128)
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        # 第四層
        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))


        # 第五層
        model.add(Dense(1, activation="sigmoid"))

        return model


    def build_D_model(self):
        """ 識別器の訓練用モデル """       

        img = Input(shape=self.img_shape)                                          # 入力画像
        y_enc = Input(shape=(self.rows, self.cols, self.d_label_channels))         # クラスラベル 
        img_y = Concatenate(axis=3)([img, y_enc])                                  # input_shape=(28,28,self.d_label_channels+1)

        discriminator = self.build_D_layers()
        cls = discriminator(img_y)

        return Model(inputs=[img, y_enc], outputs=cls)



    def build_CGAN_model(self, generator, discriminator):
        """ 生成器の訓練用モデル """

        z = Input(shape=(self.z_dim,))                                       # 潜在ベクトル      
        y_enc_g = Input(shape=(self.num_class,))                             # one-hotにエンコードしたクラスラベル for generator
        y_enc_d = Input(shape=(self.rows, self.cols, self.d_label_channels)) # 1チャンネル全体にエンコードしたクラスラベル for generator

        img = generator([z, y_enc_g])
        discriminator.trainable=False           # このモデルは生成器の訓練に使うために、識別器の重みパラメータは凍結しておく
        cls = discriminator([img, y_enc_d])

        return Model(inputs=[z, y_enc_g, y_enc_d], outputs=cls)


    def build_compile_model(self):

        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=4e-5, beta_1=0.5, beta_2=0.9)
        self.cgan_optimizer = tf.keras.optimizers.Adam(learning_rate=4e-3, beta_1=0.5, beta_2=0.9)

        self.discriminator = self.build_D_model()  # discriminatorの訓練に使う
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.d_optimizer, metrics=['accuracy'])

        self.generator = self.build_G_model()  # Generatorの訓練にはGenerator単体では使わないで、以下のようにdiscriminatorと組み合わせる

        self.cGAN = self.build_CGAN_model(self.generator, self.discriminator)  
        self.cGAN.compile(loss='binary_crossentropy', optimizer=self.cgan_optimizer, metrics=['accuracy'])


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
  
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(121)
        ax.plot(np.arange(1,len(g_hist)+1), g_hist[:,0], label="BCE loss Generator", lw=3)
        ax.plot(np.arange(1,len(d_hist)+1), d_hist[:,0], label="BCE loss Discriminator", lw=3)
        ax.set_xlabel("Iteration", size=15)
        ax.set_yscale("log")
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.legend()

        ax = fig.add_subplot(122)
        ax.plot(np.arange(1,len(g_hist)+1), g_hist[:,1], label="Accuracy Generator", lw=3)
        ax.plot(np.arange(1,len(d_hist)+1), d_hist[:,1], label="Accuracy Discriminator", lw=3)
        ax.set_xlabel("Iteration", size=15)
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.path, 'train_history.png'))


    def read_data(self):
        (X_train, y_train), (_, _) = mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
        X_train = X_train.reshape(-1, *self.img_shape)

        return X_train, y_train


    def train(self):

        X_train, y_train = self.read_data()

        g_hist = []
        d_hist = []
        imgs_hist = []

        for epoch in range(1, self.num_epoch+1):
            idx = np.random.randint(0, len(X_train), len(X_train))
            for i in range(int(X_train.shape[0] / self.batch_size)):
                itr = (epoch- 1)* int(X_train.shape[0] / self.batch_size)+ i
                
                """ 偽手書き数字の生成 """
                #z = np.random.uniform(-1, 1, (self.batch_size, self.z_dim))           # 潜在ベクトル（-1~1 均一分布）
                z = np.random.normal(0, 1, (self.batch_size, self.z_dim))              # 潜在ベクトル（μ0,σ1 正規分布）
                f_y = np.random.randint(0, self.num_class, self.batch_size)            # 数字の分類クラスをランダム生成
                f_y_enc_g = tf.keras.utils.to_categorical(f_y, self.num_class)         # 生成器に入力する分類クラスyをone-hot表示
                f_y_enc_d = np.array(list(map(self.encode_d, f_y)))                    # 識別器に入力する分類クラスyを(28x28)画素に渡って0,1表示
                f_img = self.generator([z, f_y_enc_g])                                 # 生成器で偽の手書き数字生成

                """ 本物の手書き数字の用意 """
                r_img = X_train[idx[i*self.batch_size:(i+1)*self.batch_size]]   # 実データの画像
                r_y = y_train[idx[i*self.batch_size:(i+1)*self.batch_size]]     # 実データの分類クラスy
                r_y_enc_d = np.array(list(map(self.encode_d, r_y)))             # 識別器に入力する分類クラスyを(28x28)画素に渡って0,1表示

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

                """ 識別器の訓練 """
                img = np.concatenate((r_img, f_img), axis=0)                                                 # 本物画像、偽画像の順に結合
                label = np.concatenate((r_y_enc_d, f_y_enc_d), axis=0)                                       # 本物画像に対応する数字の分類クラスy、偽画像に対応する数字の分類クラスyの順に結合
                cls = np.concatenate((np.ones((self.batch_size,1)), np.zeros((self.batch_size,1))), axis=0)  # 本物と識別で1, 偽者と識別で0とする
                d_loss = self.discriminator.train_on_batch([img, label], cls)                                # 訓練、パラメータ更新

                """ 生成器の訓練 """
                #z = np.random.uniform(-1, 1, (self.batch_size, self.z_dim))       # 潜在ベクトル（-1~1 均一分布）
                z = np.random.normal(0, 1, (self.batch_size, self.z_dim))          # 潜在ベクトル（μ0,σ1 正規分布）
                y = np.random.randint(0, self.num_class, self.batch_size)          # 数字の分類クラスをランダム生成
                y_enc_g = tf.keras.utils.to_categorical(y, self.num_class)         # 生成器に入力する分類クラスyをone-hot表示
                y_enc_d = np.array(list(map(self.encode_d, y)))                    # 識別器に入力する分類クラスyを(28x28)画素に渡って0,1表示
 
                g_loss = self.cGAN.train_on_batch([z, y_enc_g, y_enc_d], np.ones((self.batch_size,1)))       # 訓練、パラメータ更新。偽の生成画像が本物(=1)と識別されるとLossが下がる
                print(f"epoch: {epoch}, iteration: {itr}, g_loss: {g_loss[0]:.3f}, g_acc: {g_loss[1]:.3f}, d_loss: {d_loss[0]:.4f}, d_acc: {d_loss[1]:.3f}")

                """ 損失の記録 """
                g_hist.append(g_loss)
                d_hist.append(d_loss)
                
            self.generator.save_weights(os.path.join(self.path, 'generator.h5'))                   # 各エポックごとに重みパラメータを保存更新
            self.discriminator.save_weights(os.path.join(self.path, 'discriminator.h5'))           # 各エポックごとに重みパラメータを保存更新
            self.create_montage(np.array(imgs_hist), "img_history.png", (len(imgs_hist)//10, 10))  # 指定のiteration時点での画像のモンタージュをプロットし、保存更新
            self.plot_history(g_hist, d_hist)                                                      # 各エポックごとにLossとAccの訓練推移をプロットし、保存更新

if __name__ == '__main__':
    cGAN = CGAN()
    cGAN.build_compile_model()
    cGAN.train()