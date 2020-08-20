
from pathlib import Path
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


project_path = Path(__file__).resolve().parents[1]
network = 'FFNN'
model = load_model(project_path / ("models/best_" + network + ".h5"))  # rollback to best mod

plot_model(
    model,
    to_file="../img/model_" + network + "_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=900,
)
print("Finished!")