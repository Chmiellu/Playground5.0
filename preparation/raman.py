import pandas as pd
import matplotlib.pyplot as plt


file_path = 'data/Zad2_L1.csv'
df = pd.read_csv(file_path, sep=';', decimal=',', skiprows=[1])

df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric)

wavenumber = df['Wavenumber [cm^-1]']
intensities = df.drop(columns=['Wavenumber [cm^-1]'])

index_985 = (wavenumber - 985).abs().idxmin()

intensities_normalized = intensities / intensities.iloc[index_985]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(wavenumber, intensities)
ax1.set_title('Raw Raman Spectra')
ax1.set_xlabel('Wavenumber [cm^-1]')
ax1.set_ylabel('Intensity [a.u.]')

ax2.plot(wavenumber, intensities_normalized)
ax2.set_title('Normalized Raman Spectra (normalized to 985 cm^-1)')
ax2.set_xlabel('Wavenumber [cm^-1]')
ax2.set_ylabel('Normalized Intensity [a.u.]')

plt.tight_layout()
plt.show()

