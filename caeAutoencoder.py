import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def __main__():
        # Instantiate the model
        model = ConvAutoencoder()

        # Define the loss function
        criterion = nn.MSELoss()

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        num_epochs = 10
        for epoch in range(num_epochs):
            # Set the model to training mode
            model.train()
            
            # Iterate over the training dataset
            for data in train_loader:
                # Move the data to the device (e.g. GPU) if available
                data = data.to(device)
                
                # Forward pass
                output = model(data)
                
                # Compute the loss
                loss = criterion(output, data)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Print the loss for this epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")