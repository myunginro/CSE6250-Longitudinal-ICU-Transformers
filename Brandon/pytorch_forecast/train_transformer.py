import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import pandas as pd

# load data
data = pd.read_csv('train_timeseries.csv')
valid_data = pd.read_csv('validate_timeseries.csv')

# define dataset
max_encoder_length = 9999
max_prediction_length = 3
# training_cutoff = "YYYY-MM-DD"  # day for cutoff

training = TimeSeriesDataSet(
    # data[lambda x: x.date <= training_cutoff],
    data,
    time_idx=data['time_idx'],
    target=data['mortality'],
    group_ids=['group'],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    # static_categoricals=[ ... ],
    # static_reals=[ ... ],
    # time_varying_known_categoricals=[ ... ],
    # time_varying_known_reals=[ ... ],
    # time_varying_unknown_categoricals=[ ... ],
    # time_varying_unknown_reals=[ ... ],
)

validation = TimeSeriesDataSet(
    # data[lambda x: x.date <= training_cutoff],
    data,
    time_idx=data['time_idx'],
    target=data['mortality'],
    group_ids=['group'],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
)

# validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
# batch_size = 128
batch_size = 1
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)


early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    gpus=0,
    gradient_clip_val=0.1,
    limit_train_batches=30,
    callbacks=[lr_logger, early_stop_callback],
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=2,
    reduce_on_plateau_patience=4
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find optimal learning rate
res = trainer.lr_find(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

trainer.fit(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader,
)