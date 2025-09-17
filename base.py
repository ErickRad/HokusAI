import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import init, Fore
from tqdm import tqdm

init(autoreset=True)

class NeuralNetwork:
    class EncoderCNN(nn.Module):
        def __init__(self, embed_size):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)
            
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(2, 2)
            
            self.fc = nn.Linear(128 * 28 * 28, embed_size)

        def forward(self, images):
            x = self.conv1(images)
            x = F.relu(x)
            x = self.pool1(x)
            
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool2(x)
            
            x = self.conv3(x)
            x = F.relu(x)
            x = self.pool3(x)
            
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

    class DecoderRNN(nn.Module):
        def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
            super().__init__()

            self.embed = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, vocab_size)

        def forward(self, features, captions):
            embeddings = self.embed(captions[:, :-1])

            lstm_out, _ = self.lstm(embeddings)
            outputs = self.linear(lstm_out)

            return outputs

    def __init__(self, embed_size, hidden_size, vocab_size, device):
        self.device = device
        self.encoder = self.EncoderCNN(embed_size).to(device)
        self.decoder = self.DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

    def train(self, train_loader, criterion, optimizer, scheduler, num_epochs):
        self.encoder.train()
        self.decoder.train()

        for epoch in range(num_epochs):
            total_loss = 0

            loop = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{num_epochs}",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [ {elapsed}/{remaining} ,{rate_fmt} {postfix} ]"
            )

            for images, captions, lengths in loop:
                images = images.to(self.device)
                captions = captions.to(self.device)

                optimizer.zero_grad()

                features = self.encoder(images)
                outputs = self.decoder(features, captions)

                outputs = outputs.view(-1, outputs.size(2))
                targets = captions[:, 1:].contiguous().view(-1)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                loop.set_postfix(loss=f"{loss.item():.4f}")

            print(f"[{Fore.GREEN}✓{Fore.RESET}] Average loss: {total_loss/len(train_loader):.4f}\n")

    def generateCaption(self, image, vocab, max_length=10):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            image = image.to(self.device).unsqueeze(0)
            feature = self.encoder(image) 

            inputs = feature.unsqueeze(1) 
            states = None
            caption_indices = []

            for _ in range(max_length):
                lstm_out, states = self.decoder.lstm(inputs, states) 

                outputs = self.decoder.linear(lstm_out.squeeze(1)) 
                predicted = outputs.argmax(1)
                predicted_id = predicted.item()

                if predicted_id == vocab.stoi["<EOS>"]:
                    break

                caption_indices.append(predicted_id)

                inputs = self.decoder.embed(predicted)
                inputs = inputs.unsqueeze(1) 

            caption_words = [vocab.itos[idx] for idx in caption_indices if vocab.itos[idx] != "<UNK>"]
            return " ".join(caption_words)



    def save(self, encoder_path, decoder_path):
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))