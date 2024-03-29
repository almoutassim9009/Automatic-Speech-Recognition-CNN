# Modèle CNN
class SpeechCNN(torch.nn.Module):
    def __init__(self, out_features):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # couches entièrement connectées
        self.linear1 = torch.nn.Linear(in_features = 32* 7* 7, out_features = 150) # 250
        self.relu4 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features = 150, out_features = out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        #print(x.shape)

        x = x.view(-1, 32* 7* 7) 
        x = self.linear1(x)
        x = self.relu4(x)
        x = self.linear2(x)

        return x
    
    
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#input = torch.randn(1, 64, 98)
input = torch.randn(1, 64, 63)

# Instancier le modèle et le déplacer sur le GPU si disponible
model = SpeechCNN(out_features=len(Classes))

# Passe en avant
output = model(input)
#print(f"Forme de la sortie : {output.size()}")


print("*"*30)
target = torch.randn(1, len(Classes))
print("shape target :",target.shape)
print("shape input  :", input.shape)
print("*"*30)
print("shape output :", output.shape)
print("*"*30)

# Afficher le résumé du modèle
summary(model, input_size=(1, 64, 63))

from torchviz import make_dot
# Visualiser le graphique computationnel
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("model_graph", format="png")
#dot.render("model_graph")  # Enregistre le graphique au format PDF
dot



# Déplacez le modèle sur le GPU s'il est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
train_losses = []
for epoch in range(num_epochs):
    train_total_loss = 0.0  # Réinitialisation de la perte totale pour chaque époque
    for inputs, targets in train_loader:    
        # Déplacez les données d'entrée et les cibles sur le même périphérique que le modèle
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        # Calculer la perte
        loss = criterion(outputs, targets)
        train_total_loss += loss.item()
        
        # Rétropropagation et mise à jour des poids
        loss.backward()
        optimizer.step()
        # Réinitialiser les gradients
        optimizer.zero_grad()

    # Afficher la perte moyenne de l'époque
    average_loss = train_total_loss / len(train_loader)
    train_losses.append(average_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss average = [{average_loss}]")

    
    model.eval()
predictions = []
targets_list = []
n_tot_test = 0
nb_correct_test = 0

with torch.no_grad():
    for data, targets in val_loader:
        pred = model(data)
        pred_class = torch.argmax(pred, dim=1)
        # Ajouter les prédictions et les cibles aux listes
        predictions.extend(pred_class.tolist())
        targets_list.extend(targets.tolist())
        
        nb_correct_test += torch.sum(pred_class == targets)
        n_tot_test += data.shape[0]
        
    acc_test = nb_correct_test / n_tot_test    
    print(f"Sur les {n_tot_test} audios de test, {nb_correct_test} ont été reconus par le modèle")
    print(f"Accuracy test  : {acc_test.item()}")
    
    
# Tracer la courbe d'apprentissage
plt.plot(range(num_epochs), train_losses, label='Train Loss',
         color='blue', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Courbe d'apprentissage")
plt.legend()
plt.grid(True)

# Ajouter un fond de couleur
plt.savefig('Ec.png')
plt.show()


word_list = Classes
predictions_tensor = torch.tensor(predictions)
targets_tensor = torch.tensor(targets_list)
# Création d'un dictionnaire associant chaque mot à son indice
word_to_index = {word: index for index, word in enumerate(word_list)}
# Inversion de l'encodage
targets = [word_list[index] for index in targets_tensor]
predictions = [word_list[index] for index in predictions_tensor]
# Création du DataFrame
df = pd.DataFrame({'Targets': targets, 'Predictions': predictions})
# Affichage du DataFrame
df


torch.save(model.state_dict(), 'model.pth')