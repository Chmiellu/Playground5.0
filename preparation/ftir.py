import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file_path = 'data/Zad3_L1.csv'
df = pd.read_csv(file_path, sep=';', decimal=',')

wavenumber = df['Wavenumber [cm^-1]']
absorbance = df.drop(columns=['Wavenumber [cm^-1]'])

area_under_curve = np.trapz(absorbance, wavenumber, axis=0)
absorbance_normalized = absorbance.div(area_under_curve, axis=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for col in absorbance.columns:
    ax1.plot(wavenumber, absorbance[col], label=col)
ax1.set_title('Raw FTIR Spectra')
ax1.set_xlabel('Wavenumber [cm^-1]')
ax1.set_ylabel('Absorbance [a.u.]')
ax1.legend(loc='upper right')

for col in absorbance_normalized.columns:
    ax2.plot(wavenumber, absorbance_normalized[col], label=col)
ax2.set_title('Normalized FTIR Spectra')
ax2.set_xlabel('Wavenumber [cm^-1]')
ax2.set_ylabel('Normalized Absorbance [a.u.]')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()


