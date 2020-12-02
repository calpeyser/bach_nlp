How to recreate the results:

```
python code/bidir_enc_dec.py <train/pred> <latent_dim> <num_epochs> <teacher_forcing_prob>
```

For example:
1. Training the model.
```
python code/bidir_enc_dec.py train 64 200 0.75
```

2. Run prediction on the test set. This prints the error components to stdout
```
python code/bidir_enc_dec.py pred 64 200 0.75
```

This will generate an npy file with the attention maps called `attn.npy`

Then use `code/plot_attention.py` to plot the attention map.