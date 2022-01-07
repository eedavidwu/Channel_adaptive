import matplotlib.pyplot as plt
import numpy as np

x=['4 symbol','16 symbol','32 symbol','64 symbol','128 symbol']
#SNR_5=[23.06252, 24.2463, 25.33498, 26.26576, 27.09694, 27.77159, 28.30266, 28.70132, 28.986, 29.1673, 29.25914, 29.29236, 29.28406, 29.23716, 29.17372, 29.10388, 29.01975, 28.94755, 28.86807, 28.79723]
#SNR_10=[21.30101, 22.47974, 23.63021, 24.75836, 25.79884, 26.80265, 27.69012, 28.501, 29.23182, 29.86758, 30.40595, 30.8564, 31.23364, 31.52115, 31.7421, 31.91446, 32.03954, 32.12212, 32.17788, 32.21594]

SNR_orginal=[26.00,26.10,26.18,26.21,26.23]
SNR_PA =    [26.11,26.34,26.54, 26.64 ,26.70]
##64 128 PA
#0.99->1.08->1.18->1.21->1.37
#print(SNR_PA-SNR_orginal)
#SNR_attention=[33.11577, 33.11031, 33.10403]
#plt.figure(figsize=(10,10))
plt.title('Performance over symbol numbers in ratio 1/6, SNR 5')
plt.xlabel('SNR (dB)', size=10)
plt.ylabel('Ave_PSNR (dB)', size=10)
plt.plot(x, SNR_orginal, color='b', linestyle='-', marker='*',label='Traditional JSCC')
#plt.plot(x, SNR_attention_4, color='g', linestyle='-', marker='*',label='1/6, 4 symbols')
plt.plot(x, SNR_PA, color='r', linestyle='-', marker='o', label='Our model')
#nohup python train.py --S 16 --M 16 --tran_know_flag 0 --resume True --train_snr 5 --best_ckpt_path ./ckpts/SNR_5/ > symbol_16_tran_0_SNR_5.out&
# python eval.py --S 128 --M 2 --tran_know_flag 0  --train_snr 5 --best_ckpt_path ./ckpts/SNR_5/
#plt.arrow(0, 0, 0.01, np.sin(0.01), shape = 'full', lw = 10,
#   length_includes_head = True, head_width = .05, color = 'r')
plt.annotate("", xy=(x[0], SNR_orginal[0]), xytext=(x[0], SNR_PA[0]),arrowprops=dict(arrowstyle="<->",relpos=(0, 0)))
plt.annotate(' 0.11dB', xy=(x[0], SNR_orginal[0]), xytext=(x[0],  (SNR_PA[0]+SNR_orginal[0])/2))

plt.annotate("", xy=(x[1], SNR_orginal[1]), xytext=(x[1], SNR_PA[1]),arrowprops=dict(arrowstyle="<->",relpos=(0, 0)))
plt.annotate(' 0.24dB', xy=(x[1], SNR_orginal[1]), xytext=(x[1],  (SNR_PA[1]+SNR_orginal[1])/2))

plt.annotate("", xy=(x[2], SNR_orginal[2]), xytext=(x[2], SNR_PA[2]),arrowprops=dict(arrowstyle="<->",relpos=(0, 0)))
plt.annotate(' 0.36dB', xy=(x[2], SNR_orginal[2]), xytext=(x[2],  (SNR_PA[2]+SNR_orginal[2])/2))

plt.annotate("", xy=(x[3], SNR_orginal[3]), xytext=(x[3], SNR_PA[3]),arrowprops=dict(arrowstyle="<->",relpos=(0, 0)))
plt.annotate(' 0.43dB', xy=(x[3], SNR_orginal[3]), xytext=(x[3],  (SNR_PA[3]+SNR_orginal[3])/2))

plt.annotate("", xy=(x[4], SNR_orginal[4]), xytext=(x[4], SNR_PA[4]),arrowprops=dict(arrowstyle="<->",relpos=(0, 0)))
plt.annotate(' 0.47dB', xy=(x[4], SNR_orginal[4]), xytext=(x[4],  (SNR_PA[4]+SNR_orginal[4])/2))

plt.legend()
#plt.show()
plt.savefig('./PSNR_ratio_8_PA_over_symbols_SNR_5.jpg')

