import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go

# Add Favicon
st.set_page_config(
    page_title="AIVN - Advanced Gradient Vanishing Demo",
    page_icon="./static/aivn_favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

torch.cuda.empty_cache()

# Add logo
st.image("./static/aivn_logo.png", width=300)

SEED = 42
torch.manual_seed(SEED)


class MyNormalization(nn.Module):
    def __init__(self):
        super(MyNormalization, self).__init__()

    def forward(self, x):
        mean = torch.mean(x, dim=0, keepdim=True)
        std = torch.std(x, dim=0, keepdim=True)
        return (x - mean) / (std + 1e-5)


# Define a simple deep neural network
class SimpleNN(nn.Module):
    def __init__(self, num_layers=7, activation="Sigmoid", init_weights="Default", norm_type=None, use_skip=False):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.norm_type = norm_type
        self.use_skip = use_skip
        self.norm_layers = nn.ModuleList()

        # Add layers
        for _ in range(num_layers):
            self.layers.append(nn.Linear(10, 10))
            if norm_type == "BatchNorm":
                self.norm_layers.append(nn.BatchNorm1d(10))
            elif norm_type == "CustomNorm":
                self.norm_layers.append(MyNormalization())

        self.init_weights(init_weights)

    def init_weights(self, method):
        if method == "Default":
            pass
        elif method == "Increase_std=1.0":
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=1.0)
                    nn.init.constant_(module.bias, 0.0)
        elif method == "Increase_std=10.0":
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=10.0)
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        skip_connection = x
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply normalization
            if self.norm_type is not None:
                x = self.norm_layers[i](x)

            # Apply activation function
            if self.activation == "Sigmoid":
                x = torch.sigmoid(x)
            elif self.activation == "Tanh":
                x = torch.tanh(x)
            elif self.activation == "ReLU":
                x = F.relu(x)

            # Add skip connection
            if self.use_skip and i % 2 == 1:
                x = x + skip_connection
                skip_connection = x

        return x


# Streamlit app
st.title("Advanced Gradient Vanishing Demo")

st.sidebar.header("Settings")

# Number of layers
# num_layers = st.sidebar.slider(
# "Number of Layers", min_value=3, max_value=30, value=10, step=1)
num_layers = 7

# Activation function
activation = st.sidebar.selectbox(
    "Activation Function", ["Sigmoid", "Tanh", "ReLU"]
)

# Weight initialization
init_weights = st.sidebar.selectbox(
    "Weight Initialization", [
        "Default", "Increase_std=1.0", "Increase_std=10.0"]
)

# Normalization
norm_type = st.sidebar.selectbox(
    "Normalization Type", ["None", "BatchNorm", "CustomNorm"]
)

# Skip connection
use_skip = st.sidebar.checkbox("Use Skip Connection")

# Optimizer
optimizer_type = st.sidebar.selectbox(
    "Optimizer", ["SGD", "Adam"]
)

# Number of epochs
num_epochs = st.sidebar.slider(
    "Number of Epochs", min_value=1, max_value=100, value=10)

# Learning rate
learning_rate = st.sidebar.select_slider(
    "Learning Rate", options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0], value=1e-3
)

# Input data distribution
# input_type = st.sidebar.selectbox(
#     "Input Data Distribution", ["Gaussian (Normal)", "Uniform"]
# )
input_type = "Gaussian (Normal)"

# Batch size
# batch_size = st.sidebar.slider(
#     "Batch Size", min_value=1, max_value=100, value=32)
batch_size = 32

# Generate input data
if input_type == "Gaussian (Normal)":
    input_data = torch.randn(batch_size, 10)
else:  # Uniform
    input_data = torch.rand(batch_size, 10)

# Create model
model = SimpleNN(num_layers=num_layers, activation=activation, init_weights=init_weights,
                 norm_type=norm_type if norm_type != "None" else None, use_skip=use_skip)

# Select optimizer
if optimizer_type == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
elif optimizer_type == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Forward pass to compute activations
output = model(input_data)

# Compute gradients
criterion = nn.MSELoss()  # Use Mean Squared Error
target = torch.zeros_like(output)  # Dummy target to compute the loss

# Train model for multiple epochs
losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Zero the gradients
    output = model(input_data)  # Forward pass
    loss = criterion(output, target)  # Calculate loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model weights
    losses.append(loss.item())  # Track the loss value

# Extract gradients
gradients = []
for i, layer in enumerate(model.layers):
    if layer.weight.grad is not None:
        gradients.append(layer.weight.grad.norm().item())
    else:
        gradients.append(0)


# Interactive plot using Plotly
fig = go.Figure()

# Gradient plot with annotations
fig.add_trace(
    go.Scatter(
        x=list(range(1, len(gradients) + 1)),
        y=gradients,
        mode="lines+markers+text",
        text=[f"{g:.4f}" for g in gradients],
        textposition="top center",
        name="Gradient Mean",
        line=dict(color="blue"),
    )
)

fig.update_layout(
    title="Gradient Mean Across 7 Layers",
    xaxis_title="Layer",
    yaxis_title="Gradient Mean",
    legend=dict(yanchor="top", y=1, xanchor="right", x=1),
    hovermode="x unified",
)

st.plotly_chart(fig)

# Observations
st.markdown("### Observations:")
st.markdown(
    f"""
    **Configuration:**
    - **Activation Function:** `{activation}`
    - **Weight Initialization:** `{init_weights}`
    - **Normalization Type:** `{norm_type}`
    - **Optimizer:** `{optimizer_type}`
    - **Learning Rate:** `{learning_rate}`
    - **Skip Connection:** `{"Enabled" if use_skip else "Disabled"}`
    
    **Gradient Insights:**
    - Total Layers: **{num_layers}**
    - Minimum Gradient Mean: **{min(gradients):.4f}**
    - Maximum Gradient Mean: **{max(gradients):.4f}**
    - Gradient Means displayed above for each layer.
    """
)

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        color: #555;
    }
    </style>
    <div class="footer">
        2024 AI VIETNAM | Made by <a href="https://github.com/Koii2k3/GradientVanishing" target="_blank">Koii2k3</a>
    </div>
    """,
    unsafe_allow_html=True
)
