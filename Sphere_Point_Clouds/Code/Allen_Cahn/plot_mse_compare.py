import os
import pandas as pd
import matplotlib.pyplot as plt



plt.rcParams.update({'font.size': 12})


T_all = [3, 4, 5, 6]
embed_dim = 7
eps = 0.001
weight_decay = 1e-2
num_layers = 4
scale_factor = 1
hard_thresholding_fraction = 1
nepochs = 100
lr = 1e-4

base_dir_type1 = '/home/qianhou/torch-harmonics/notebooks/Allen_Canh/Nufft3D_Test/AC_sphere_infer/Nufft3d_embed{}/Infer_T{}/eps{}_alpha{}_AC_layer{}_sf{}/ks3_ratio{}_epoch{}_lr{}'
base_dir_type2 = '/home/qianhou/torch-harmonics/notebooks/Allen_Canh/Nufft3D_Test/AC_sphere_infer/Nufft3d_Type2_embed{}/Infer_T{}/eps{}_alpha{}_AC_layer{}_sf{}/ks3_ratio{}_epoch{}_lr{}'
base_dir_lsfno = '/home/qianhou/torch-harmonics/notebooks/Allen_Canh/LSFNO_infer/LSFNO_embed{}/Infer_T{}/eps{}_alpha{}_AC_layer{}_sf{}/ks3_ratio{}_epoch{}_lr{}'

=
def plot_loss(df1, df2, df3, T, start_epoch=0, suffix="all"):
    plt.figure(figsize=(10, 6))
    plt.plot(df1['Epoch'][start_epoch:], df1['Training Loss'][start_epoch:], label='LSR-Type1')
    plt.plot(df2['Epoch'][start_epoch:], df2['Training Loss'][start_epoch:], label='LSR-Type2')
    plt.plot(df3['Epoch'][start_epoch:], df3['Training Loss'][start_epoch:], label='LSFNO')

    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss vs Epoch (T={T}, Epoch {start_epoch}~{nepochs})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    save_dir = f'./training_loss_figs/T{T}/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'training_loss_T{T}_{suffix}.png'), dpi=300)
    plt.close()

for T in T_all:

    path1 = base_dir_type1.format(embed_dim, T, eps, weight_decay, num_layers, scale_factor,
                                   hard_thresholding_fraction, nepochs, lr) + '/training_loss.csv'
    path2 = base_dir_type2.format(embed_dim, T, eps, weight_decay, num_layers, scale_factor,
                                   hard_thresholding_fraction, nepochs, lr) + '/training_loss.csv'
    path3 = base_dir_lsfno.format(embed_dim, T, eps, weight_decay, num_layers, scale_factor,
                                   hard_thresholding_fraction, nepochs, lr) + '/training_loss.csv'


    df_type1 = pd.read_csv(path1)
    df_type2 = pd.read_csv(path2)
    df_lsfno = pd.read_csv(path3)


    plot_loss(df_type1, df_type2, df_lsfno, T=T, start_epoch=0, suffix="all")

  
    plot_loss(df_type1, df_type2, df_lsfno, T=T, start_epoch=50, suffix="epoch_gt50")
