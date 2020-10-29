import matplotlib.pyplot as plt

def plot_attention(attn_energy_matrixes):
    mats = []
    dec_inputs = []
    for dec_ind, attn in enumerate(attn_energy_matrixes[0]):
        mats.append(attn.reshape(-1))
        dec_inputs.append(dec_ind)
    mats = np.array(mats)

    fig, ax = plt.subplots(figsize=(32, 32))
    ax.set_xlabel('chorale_slice')
    ax.set_ylabel('rna_chord')
    ax.imshow(mats)
