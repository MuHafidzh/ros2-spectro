# Joystick Controls untuk Spectrometer

## Mapping Tombol Joystick

| Tombol              | Fungsi                          | Keterangan                                      |
| ------------------- | ------------------------------- | ----------------------------------------------- |
| X (Button 0)        | Save Intensity Data             | Menyimpan data intensitas ke file TXT           |
| Triangle (Button 2) | Save Reflectance Data           | Menyimpan data reflectance ke file TXT          |
| Start (Button 10)   | Toggle Tab Mode                 | Beralih antara tab Intensity dan Reflectance    |
| PS (Button 9)       | Connect/Disconnect Spectrometer | Menghubungkan atau memutus koneksi spectrometer |

## Fitur UI

### Status Koneksi Spectrometer

- Ditampilkan di bagian atas UI: **Status: Connected** (hijau) atau **Status: Disconnected** (merah)
- Otomatis terupdate saat connect/disconnect via PS button
- Plot akan di-clear otomatis saat disconnect untuk menghindari data lama

### Anti-Save saat Disconnect

- Tombol X dan Triangle akan menampilkan peringatan jika spectrometer tidak terkoneksi
- Mencegah penyimpanan data yang tidak valid

## Fitur Anti-Bouncing

- Semua tombol memiliki debounce time 300ms untuk mencegah multiple trigger
- Tombol hanya akan aktif pada transisi dari tidak ditekan ke ditekan (edge detection)

## Parameter Launch File

```python
{'joystick_intensity_button': 0},    # X button
{'joystick_reflectance_button': 2},  # Triangle button
{'joystick_ps_button': 10},          # PS button
{'joystick_start_button': 9},        # Start button
```

## Struktur Folder Penyimpanan

Data akan disimpan dengan struktur folder terpisah:

```
~/spectro_data/
├── intensity/
│   ├── 20250910_123456_789.txt
│   ├── sample1_20250910_123500_123.txt
│   └── ...
├── reflectance/
│   ├── 20250910_123457_456.txt
│   ├── sample1_20250910_123501_789.txt
│   └── ...
└── ...
```

- Jika field `savepath` kosong: nama file hanya timestamp
- Jika field `savepath` diisi: nama file menjadi `{nama}_{timestamp}.txt`

## Catatan

- Tombol A (button 0) telah dihapus untuk menghindari duplikasi fungsi
- Toggle tab hanya bekerja jika reflectance tab sudah di-enable
- PS button berguna untuk reconnect spectrometer jika terjadi error koneksi
