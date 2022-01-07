import matplotlib.pyplot as plt
import numpy as np

x=['4 symbol','16 symbol','32 symbol','64 symbol','128 symbol']
SNR_orginal=[28.86,29.32,29.65,29.76,29.82]
SNR_PA =    [29.27,29.79,30.21,30.51,30.65]
#0.41->0.47->0.56->0.75->0.83

#print(SNR_PA-SNR_orginal)
#SNR_attention=[33.11577, 33.11031, 33.10403]
plt.title('Performance over symbol numbers in ratio 1/6, SNR 15')
plt.xlabel('SNR (dB)', size=10)
plt.ylabel('Ave_PSNR (dB)', size=10)
plt.plot(x, SNR_orginal, color='b', linestyle='-', marker='*',label='Traditional JSCC')
#plt.plot(x, SNR_attention_4, color='g', linestyle='-', marker='*',label='1/6, 4 symbols')
plt.plot(x, SNR_PA, color='r', linestyle='-', marker='o', label='Our model')
plt.annotate("", xy=(x[0], SNR_orginal[0]), xytext=(x[0], SNR_PA[0]),arrowprops=dict(arrowstyle="<->",relpos=(0, 0)))
plt.annotate(' 0.41dB', xy=(x[0], SNR_orginal[0]), xytext=(x[0],  (SNR_PA[0]+SNR_orginal[0])/2))

plt.annotate("", xy=(x[1], SNR_orginal[1]), xytext=(x[1], SNR_PA[1]),arrowprops=dict(arrowstyle="<->",relpos=(0, 0)))
plt.annotate(' 0.47dB', xy=(x[1], SNR_orginal[1]), xytext=(x[1],  (SNR_PA[1]+SNR_orginal[1])/2))

plt.annotate("", xy=(x[2], SNR_orginal[2]), xytext=(x[2], SNR_PA[2]),arrowprops=dict(arrowstyle="<->",relpos=(0, 0)))
plt.annotate(' 0.56dB', xy=(x[2], SNR_orginal[2]), xytext=(x[2],  (SNR_PA[2]+SNR_orginal[2])/2))

plt.annotate("", xy=(x[3], SNR_orginal[3]), xytext=(x[3], SNR_PA[3]),arrowprops=dict(arrowstyle="<->",relpos=(0, 0)))
plt.annotate(' 0.75dB', xy=(x[3], SNR_orginal[3]), xytext=(x[3],  (SNR_PA[3]+SNR_orginal[3])/2))

plt.annotate("", xy=(x[4], SNR_orginal[4]), xytext=(x[4], SNR_PA[4]),arrowprops=dict(arrowstyle="<->",relpos=(0, 0)))
plt.annotate(' 0.83dB', xy=(x[4], SNR_orginal[4]), xytext=(x[4],  (SNR_PA[4]+SNR_orginal[4])/2))

#plt.ylim([28, 31])
plt.legend()
plt.show()
plt.savefig('./PSNR_ratio_8_PA_over_symbols_SNR_15.jpg')

