import os
import glob
import json
import re
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

def parse_resume_file(filepath):
    with open(filepath, 'r') as f:
        contents = f.read()

    pattern = (r'epoch\s+(\d+)\s*[\r\n]+'
               r'-\s*train loss:\s*([\d.]+)[\r\n]+'
               r'-\s*test loss:\s*([\d.]+)[\r\n]+'
               r'-\s*train accuracy:\s*([\d.]+)[\r\n]+'
               r'-\s*test accuracy:\s*([\d.]+)')
    epochs = re.findall(pattern, contents)
    if not epochs:
        return []

    records = []
    for epoch_data in epochs:
        record = {
                "epoch": int(epoch_data[0]),
                "train_loss": float(epoch_data[1]),
                "test_loss": float(epoch_data[2]),
                "train_accuracy": float(epoch_data[3]),
                "test_accuracy": float(epoch_data[4])
                }
        records.append(record)
    return records

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        return {}

def load_run_data():
    run_folders = [f for f in os.listdir(".") if os.path.isdir(f)]

    all_records = []
    for folder in run_folders:
        resume_path = os.path.join(folder, "training_resume.txt")
        config_path = os.path.join(folder, "config.json")

        if not os.path.isfile(resume_path) or not os.path.isfile(config_path):
            continue

        config = load_config(config_path)
        epoch_records = parse_resume_file(resume_path)
        if not epoch_records:
            continue

        folder_parts = folder.split("_")
        variant_from_folder = folder_parts[-1] if folder_parts else ""

        for rec in epoch_records:
            record = {
                    "folder": folder,
                    "epoch": rec["epoch"],
                    "train_loss": rec["train_loss"],
                    "test_loss": rec["test_loss"],
                    "train_accuracy": rec["train_accuracy"],
                    "test_accuracy": rec["test_accuracy"],
                    "model": config.get("model", folder_parts[0] if len(folder_parts) > 0 else ""),
                    "preprocessing": config.get("preprocessing", folder_parts[1] if len(folder_parts) > 1 else ""),
                    "optimizer": config.get("optimizer", {}).get("name", folder_parts[2] if len(folder_parts) > 2 else ""),
                    "loss_function": config.get("loss", folder_parts[3] if len(folder_parts) > 3 else ""),
                    "scheduler": config.get("scheduler", {}).get("name", folder_parts[4] if len(folder_parts) > 4 else ""),
                    "variant": config.get("paths", {}).get("train_dir", "").split("/")[1] if config.get("paths", {}).get("train_dir", "") else variant_from_folder
                    }
            all_records.append(record)
    if not all_records:
        print("No valid training runs found in any folder.")
    return pd.DataFrame(all_records)

# Load the data (this assumes your run folders are in the current directory)
df = load_run_data()
df["combo"] = df.apply(lambda row: f"{row['model']}, {row['preprocessing']}, {row['optimizer']}, {row['loss_function']}, {row['scheduler']}, {row['variant']}", axis=1)
df = df.sort_values(["folder", "epoch"])

# Get unique combos for filtering
unique_combos = sorted(df["combo"].unique())

# Create interactive Dash app
app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='config-dropdown',
            options=[{'label': combo, 'value': combo} for combo in unique_combos],
            value=unique_combos,  # default is all selected
            multi=True
            )
        ]),
    html.Br(),
    dcc.Graph(id='loss-graph'),
    html.Br(),
    dcc.Graph(id='accuracy-graph')
    ])

@app.callback(
        [Output('loss-graph', 'figure'), Output('accuracy-graph', 'figure')],
        [Input('config-dropdown', 'value')]
        )
def update_graphs(selected_combos):
    filtered_df = df[df["combo"].isin(selected_combos)]

    loss_traces = []
    acc_traces = []

    for combo in selected_combos:
        sub_df = filtered_df[filtered_df["combo"] == combo]
        sub_df = sub_df.sort_values("epoch")
        loss_traces.append(
                go.Scatter(
                    x=sub_df["epoch"],
                    y=sub_df["train_loss"],
                    mode="lines+markers",
                    name=f"{combo} (Train Loss)"
                    )
                )
        loss_traces.append(
                go.Scatter(
                    x=sub_df["epoch"],
                    y=sub_df["test_loss"],
                    mode="lines+markers",
                    name=f"{combo} (Test Loss)"
                    )
                )
        acc_traces.append(
                go.Scatter(
                    x=sub_df["epoch"],
                    y=sub_df["train_accuracy"],
                    mode="lines+markers",
                    name=f"{combo} (Train Acc)"
                    )
                )
        acc_traces.append(
                go.Scatter(
                    x=sub_df["epoch"],
                    y=sub_df["test_accuracy"],
                    mode="lines+markers",
                    name=f"{combo} (Test Acc)"
                    )
                )

    loss_layout = go.Layout(
            title="Training and Test Loss Curves",
            xaxis=dict(title="Epoch"),
            yaxis=dict(title="Loss"),
            hovermode="closest"
            )
    acc_layout = go.Layout(
            title="Training and Test Accuracy Curves",
            xaxis=dict(title="Epoch"),
            yaxis=dict(title="Accuracy"),
            hovermode="closest"
            )

    loss_fig = go.Figure(data=loss_traces, layout=loss_layout)
    acc_fig = go.Figure(data=acc_traces, layout=acc_layout)
    return loss_fig, acc_fig

if __name__ == "__main__":
    app.run(debug=True)
