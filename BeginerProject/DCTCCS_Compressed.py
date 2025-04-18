import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# === 1. Nhập dữ liệu ===
N = 500
x = np.array([1182,1188,1178,1172,1125,1099,1088,1051,1034,1018,995,945,959,899,901,919,880,809,813,818,854,831,802,801,813,839,900,903,896,911,926,944,932,962,961,970,934,945,976,983,995,959,967,977,963,959,976,986,1005,991,1011,1002,996,993,962,945,880,801,783,806,811,807,819,800,815,816,816,828,826,827,818,779,810,865,862,878,879,887,876,895,915,900,871,880,903,913,881,889,889,886,882,862,864,873,897,912,896,877,864,903,897,896,897,870,897,902,915,919,906,918,927,1046,1126,1121,1117,1136,1119,1139,1148,1166,1183,1188,1162,1183,1182,1187,1213,1210,1197,1217,1212,1201,1193,1168,1162,1178,1199,1163,1156,1163,1134,1117,1022,983,1019,1092,1110,1105,1101,1115,1094,1133,1109,1129,1132,1121,1109,1116,1101,1099,1104,1095,1101,1099,1100,1091,1107,1097,1129,1127,1123,1090,1084,1008,967,885,841,870,891,911,927,1037,1142,1287,1455,1622,1838,1965,2077,2138,2075,1786,1581,1415,1294,1193,1106,1040,964,948,954,928,944,925,900,880,857,881,862,886,865,907,898,934,923,915,891,897,923,912,924,925,915,931,960,927,878,811,852,895,931,922,958,963,959,1003,1003,989,1012,992,992,987,1011,994,1022,1023,1056,1040,1046,1011,1006,1009,1017,1005,1020,1002,1030,1039,1038,1035,1054,1089,1059,1075,1071,1066,1071,1059,1038,942,912,912,951,1008,1072,1102,1101,1123,1136,1154,1151,1143,1140,1150,1171,1184,1207,1232,1203,1168,1147,1136,1136,1150,1121,1119,1088,1101,1099,1107,1104,1104,1099,1101,1073,1050,1054,1003,1002,975,915,845,847,897,942,912,906,807,688,621,523,528,581,679,742,769,788,801,798,816,848,839,835,866,889,883,891,894,899,880,849,843,853,881,910,914,935,927,971,959,968,976,997,943,946,938,923,952,953,960,979,1002,990,976,978,976,982,977,977,1004,1009,1003,1014,994,1013,1008,978,1040,1086,1161,1181,1186,1215,1205,1238,1246,1254,1278,1275,1280,1275,1281,1234,1195,1119,1088,1047,1009,1009,989,965,959,939,892,940,953,986,982,976,1008,1007,983,977,994,976,1008,1042,1038,1015,1054,1043,1049,1033,1072,1072,1050,1071,1063,1088,1077,1046,1065,1035,1047,1031,1048,1059,1039,1031,1034,1031,1023,1059,1047,1041,1042,1059,1055,1073,1056,1071,1062,1077,1070,1072,1072,1046,1047,998,993,1035,1106,1187,1233,1219,1242,1215,1232,1240,1205,1169,1142,1087,1040,1067,1067,1059,1052,1049,1102,1217,1375,1666,1920,2045,2063,2165,2127,1861,1598,1416,1293,1227,1136,1074,1034,1042,1059,1033,1053], dtype=np.float32)


# === 2. Kết quả DCT từ ESP32 ===
# Ví dụ: lấy từ Serial Monitor và dán vào đây
X_esp32 = np.array([
    23210.96,-954.59,188.69,-1843.47,387.14,64.56,860.66,528.38,755.41,-1226.50,749.09,451.81,-596.72,-548.10,5.17,189.95,810.48,495.71,-318.54,-601.06,293.73,1071.35,-180.52,456.14,-215.56,334.26,679.54,553.23,-459.52,-236.10,-594.87,881.81,85.62,511.05,-880.76,-8.20,97.37,785.61,-198.61,286.88,-982.27,502.48,219.74,635.54,-651.23,-70.92,-346.82,438.78,-44.44,214.15,-648.03,-57.15,109.38,342.92,252.69,-380.70,-304.27,-49.88,473.79,39.88,242.39,-390.56,60.69,-56.91,530.14,-288.89,17.27,-408.48,269.74,11.00,230.14,-327.86,81.00,-79.92,251.67,-103.39,207.02,-234.47,34.13,89.20,297.58,-193.03,-23.06,-146.01,27.10,-44.75,136.70,-182.52,-52.68,107.19,28.99,-9.88,-62.79,-45.89,49.11,239.23,-144.66,28.64,-54.02,39.16,-90.45,176.92,-87.60,-43.98,-101.41,187.34,-105.11,92.58,-0.69,-67.74,-17.31,65.29,-57.49,27.45,-55.49,30.75,63.10,107.17,-90.59,-102.55,-24.68,18.74,13.26,65.93,-22.89,-65.23,22.05,19.03,-12.78,-24.99,56.09,-60.85,54.15,9.54,6.91,-53.92,37.32,-51.98,17.29,-1.55,-41.99,-72.46,115.27,47.29,-46.64,18.12,-7.81,29.48,25.80,60.91,12.42,18.01,-8.55,5.42,-54.01,-70.81,-13.97,-3.18,47.95,-42.05,31.05,-41.03,-22.28,-12.21,36.80,-45.36,-42.38,-59.80,56.72,-5.92,-8.66,-53.49,-31.97,-24.45,17.17,-20.19,-24.27,13.15,61.51,44.14,-23.37,-43.69,-6.44,1.42,18.05,23.95,9.15,24.47,-7.08,-1.85,27.22,37.43,-23.89,-33.47,-12.73,31.53,-34.78,55.02,-7.39,40.91,15.58,50.93,-51.01,5.25,-18.55,40.81,-17.30,17.35,-57.45,-6.17,-24.37,-6.96,-13.70,28.49,-4.83,-3.25,-33.82,-18.95,-35.84,20.46,-26.74,19.83,-1.83,-28.42,-12.14,-10.57,15.27,8.90,-32.94,-0.59,21.60,2.78,37.42,-6.51,8.64,1.24,27.51,-37.61,0.68,-36.53,19.36,-14.24,26.75,-30.92,-9.85,-30.53,17.22,0.00,48.56,-6.62,14.18,-12.09,24.56,-1.60,14.66,-21.04,-1.03,29.02,22.39,15.79,-10.39,14.25,-2.36,-4.93,-5.59,-15.23,-18.46,-0.26,-0.35,30.99,-30.27,20.48,-25.50,13.36,-11.91,24.14,-17.70,-27.92,-13.43,-1.84,-15.24,-2.68,-38.04,10.03,31.22,11.68,-15.99,0.04,9.41,-15.46,6.54,-5.23,-24.05,12.87,6.77,2.35,13.74,-11.40,9.84,-3.04,17.91,0.06,17.16,6.87,8.06,-3.02,14.83,-13.17,-18.77,6.51,35.92,-22.22,11.36,-12.36,24.44,-7.29,19.84,-8.99,7.54,-7.71,7.77,13.39,19.52,1.13,14.11,-1.66,13.13,-7.14,-23.35,3.19,22.34,26.54,3.03,-10.28,-25.41,4.13,2.83,7.36,13.94,18.27,-14.69,23.62,-15.16,10.75,-2.55,-16.71,-6.14,-6.89,5.83,-9.10,-14.11,-17.66,-1.32,-1.52,5.38,-6.23,3.52,-5.29,-8.10,-21.24,-12.83,-5.24,-20.24,6.06,-11.29,-5.00,10.03,-21.79,0.97,16.11,-4.40,3.90,2.61,15.63,3.22,6.25,7.32,-28.93,2.50,-1.41,-0.02,-6.66,18.71,-8.50,10.31,-0.81,3.44,-8.96,-4.30,20.31,7.78,-5.27,3.13,-13.88,16.20,14.53,0.18,-11.58,13.00,-14.96,7.26,4.27,3.58,-4.54,-11.53,-23.07,-3.57,-28.49,9.97,-13.67,20.93,15.44,-10.76,14.01,-9.30,-1.77,-0.50,10.69,-11.03,1.84,9.79,3.86,-1.61,-12.58,10.46,5.00,-17.64,-21.68,-15.17,21.17,1.28,-6.42,-5.29,-9.17,0.89,-5.00,1.40,-9.65,-19.76,4.65,12.90,-5.55,-4.39,-11.78,-7.43,15.08,-6.82,4.36,6.94,-20.14,22.39,10.55,-3.92,-3.83,2.05,2.02,-20.37,6.49,-6.18,-7.23,-9.59,-19.16,10.48,5.00,-1.17,-25.98,-10.47,16.26,21.41,-7.23,10.39,-17.33,-7.92,12.91,9.43,5.88,-0.42,29.46,-7.82,13.52,-7.04,6.52,19.46,11.16,-10.86,-11.40,3.12,22.20,7.31,-11.83,-3.64,15.18,-5.52,9.86,14.07,9.78,0.14,-14.78,-4.58,6.88

    # ... (đủ 500 giá trị)
], dtype=np.float32)

# === 3. Tạo LUT giống ESP32 ===
DCT_LUT = np.zeros((N, N), dtype=np.float32)
for k in range(N):
    for n in range(N):
        DCT_LUT[k, n] = np.cos(np.pi * (n + 0.5) * k / N)

DCT_LUT *= np.sqrt(2 / N)
DCT_LUT[0, :] *= 1 / np.sqrt(2)

# === 4. Tính DCT bằng LUT ===
X_lut = np.zeros(N, dtype=np.float32)
for k in range(N):
    X_lut[k] = np.dot(x, DCT_LUT[k])

# === 5. DCT chuẩn bằng SciPy ===
X_ref = dct(x, type=2, norm='ortho')

# === 6. Vẽ biểu đồ so sánh ===
plt.figure(figsize=(14, 6))
plt.plot(X_ref, label='SciPy DCT', linewidth=2)
plt.plot(X_lut, '--', label='DCT LUT (Python)', linewidth=1.5)
if len(X_esp32) == N:
    plt.plot(X_esp32, ':', label='ESP32 DCT', linewidth=1.5)
else:
    print("⚠️ ESP32 DCT chưa đủ 500 phần tử! Đang có:", len(X_esp32))

plt.title('So sánh DCT: SciPy vs LUT Python vs ESP32')
plt.xlabel('Tần số (Frequency Index)')
plt.ylabel('Biên độ (Amplitude)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 7. Sai số (nếu ESP32 đủ 500 phần tử) ===
if len(X_esp32) == N:
    print("Sai số ESP32 vs SciPy: trung bình =", np.mean(np.abs(X_esp32 - X_ref)))
    print("Sai số ESP32 vs LUT Python: trung bình =", np.mean(np.abs(X_esp32 - X_lut)))

# === 8. DCT loại 3 bằng SciPy ===
X_ref_type3 = dct(X_ref, type=3, norm='ortho')

# === 9. Vẽ biểu đồ DCT loại 3 ===
plt.figure(figsize=(14, 6))
plt.plot(X_ref_type3, label='SciPy DCT Type 3', linewidth=2)
plt.title('DCT Type 3 (SciPy)')
plt.xlabel('Tần số (Frequency Index)')
plt.ylabel('Biên độ (Amplitude)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 10. Tính toán MSE, PSNR, SSIM ===
from skimage.metrics import structural_similarity as ssim

if len(X_esp32) == N:
    mse = np.mean((X_esp32 - X_ref) ** 2)
    psnr = 10 * np.log10(np.max(X_ref) ** 2 / mse)
    ssim_value = ssim(X_esp32, X_ref, data_range=X_ref.max() - X_ref.min())

    print("MSE (ESP32 vs SciPy):", mse)
    print("PSNR (ESP32 vs SciPy):", psnr, "dB")
    print("SSIM (ESP32 vs SciPy):", ssim_value)

# === 11. Hiển thị 4 biểu đồ cùng lúc ===
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Biểu đồ 1: DCT loại 2
axs[0, 0].plot(X_ref, label='SciPy DCT Type 2', linewidth=2)
axs[0, 0].set_title('DCT Type 2 (SciPy)')
axs[0, 0].set_xlabel('Tần số (Frequency Index)')
axs[0, 0].set_ylabel('Biên độ (Amplitude)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Biểu đồ 2: DCT loại 3
axs[0, 1].plot(X_ref_type3, label='SciPy DCT Type 3', linewidth=2)
axs[0, 1].set_title('DCT Type 3 (SciPy)')
axs[0, 1].set_xlabel('Tần số (Frequency Index)')
axs[0, 1].set_ylabel('Biên độ (Amplitude)')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Biểu đồ 3: MSE
if len(X_esp32) == N:
    mse_values = np.abs(X_esp32 - X_ref) ** 2
    axs[1, 0].plot(mse_values, label='MSE (ESP32 vs SciPy)', linewidth=2)
    axs[1, 0].set_title('Mean Squared Error (MSE)')
    axs[1, 0].set_xlabel('Tần số (Frequency Index)')
    axs[1, 0].set_ylabel('MSE Value')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

# Biểu đồ 4: PSNR
if len(X_esp32) == N:
    psnr_values = 10 * np.log10(np.max(X_ref) ** 2 / mse_values)
    axs[1, 1].plot(psnr_values, label='PSNR (ESP32 vs SciPy)', linewidth=2)
    axs[1, 1].set_title('Peak Signal-to-Noise Ratio (PSNR)')
    axs[1, 1].set_xlabel('Tần số (Frequency Index)')
    axs[1, 1].set_ylabel('PSNR (dB)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

# === 12. Vẽ chồng các biểu đồ để so sánh ===
plt.figure(figsize=(14, 8))

# DCT loại 2 (SciPy)
plt.plot(X_ref, label='SciPy DCT Type 2', linewidth=2)

# DCT loại 3 (SciPy)
plt.plot(X_ref_type3, '--', label='SciPy DCT Type 3', linewidth=1.5)

# ESP32 DCT
if len(X_esp32) == N:
    plt.plot(X_esp32, ':', label='ESP32 DCT', linewidth=1.5)

# LUT DCT
plt.plot(X_lut, '-.', label='DCT LUT (Python)', linewidth=1.5)

# Cấu hình biểu đồ
plt.title('So sánh chồng: DCT Type 2, DCT Type 3, ESP32 DCT, LUT DCT')
plt.xlabel('Tần số (Frequency Index)')
plt.ylabel('Biên độ (Amplitude)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 13. Biểu đồ chi tiết: Biên dạng gốc, tái tạo, DCT trước/sau nén, sai lệch ===
compression_ratio = 0.5  # Giả sử giữ lại 50% hệ số DCT
num_coefficients = int(N * compression_ratio)

# Nén DCT bằng cách giữ lại một số hệ số đầu tiên
dct_data = X_ref
dct_data_compressed = np.zeros_like(dct_data)
dct_data_compressed[:num_coefficients] = dct_data[:num_coefficients]

# Tái tạo tín hiệu từ DCT nén
reconstructed = dct(dct_data_compressed, type=3, norm='ortho')

# Tính toán MSE, PSNR, SSIM
mse = np.mean((x - reconstructed) ** 2)
psnr = 10 * np.log10(np.max(x) ** 2 / mse)
ssim_val = ssim(x, reconstructed, data_range=x.max() - x.min())

# Vẽ các biểu đồ
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Biên dạng gốc và tái tạo
axs[0, 0].plot(x, label='Gốc')
axs[0, 0].plot(reconstructed, label='Tái tạo', linestyle='--')
axs[0, 0].set_title(f"Miền thời gian\nMSE={mse:.2f}, PSNR={psnr:.2f} dB, SSIM={ssim_val:.4f}")
axs[0, 0].legend()

# Hệ số DCT gốc
axs[0, 1].stem(dct_data, linefmt='C1-', markerfmt='C1o', basefmt=" ")
axs[0, 1].set_title("Hệ số DCT (trước nén)")

# Hệ số DCT sau nén
axs[1, 0].stem(dct_data_compressed, linefmt='C2-', markerfmt='C2o', basefmt=" ")
axs[1, 0].set_title(f"Hệ số DCT (sau khi giữ {compression_ratio*100:.0f}%)")

# Độ sai lệch
axs[1, 1].plot(x - reconstructed, color='red')
axs[1, 1].set_title("Sai lệch (Original - Reconstructed)")

plt.tight_layout()
plt.show()
