from wfmatching import wfmatching

print("Near-field pixel scale:", lf.microns_per_pixel, "um/pixel")
print("Nyquist frequency:", 1 / (2 * lf.microns_per_pixel), "cycles/um")
print("FFT frequency spacing:", fx[1] - fx[0], "cycles/um")

plt.imshow(
    np.abs(far_field) ** 2,
    extent=[fx[0], fx[-1], fy[0], fy[-1]],
    origin="lower",
)
plt.xlabel("fx [cycles/um]")
plt.ylabel("fy [cycles/um]")
