import joblib
import numpy as np
import pandas as pd
import math
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


# Reading data
game2idx = joblib.load('/Users/a1234/Desktop/workspace/Steam_Recommendation_System_777_tp/Processed_Data/model_pkl_small_dataset/game2idx.pkl')
idx2game = joblib.load('/Users/a1234/Desktop/workspace/Steam_Recommendation_System_777_tp/Processed_Data/model_pkl_small_dataset/idx2game.pkl')
rec = joblib.load('/Users/a1234/Desktop/workspace/Steam_Recommendation_System_777_tp/Processed_Data/model_pkl_small_dataset/rec.pkl')
hours = joblib.load('/Users/a1234/Desktop/workspace/Steam_Recommendation_System_777_tp/Processed_Data/model_pkl_small_dataset/hours.pkl')
buy = joblib.load('/Users/a1234/Desktop/workspace/Steam_Recommendation_System_777_tp/Processed_Data/model_pkl_small_dataset/buy.pkl')
users = joblib.load('/Users/a1234/Desktop/workspace/Steam_Recommendation_System_777_tp/Processed_Data/model_pkl_small_dataset/buyers.pkl')

# List of game names
gamelist = list(game2idx)

# Number of games
n_game = len(gamelist)

# Passing dictionary
gamedict = {1:"NULL", 2:"NULL", 3:"NULL", 4:"NULL", 5:"NULL"}
timedict = {1:"NULL", 2:"NULL", 3:"NULL", 4:"NULL", 5:"NULL"}
idxdict = {1:"NULL", 2:"NULL", 3:"NULL", 4:"NULL", 5:"NULL"}

# The following two are to be passed
usertime = []
useridx = []

# Recommended games to return
recgame = []


# User Similarity Recommendation
def UserSimilarity(games, game_hours):
    similarity = np.zeros(len(users))  # User similarity matrix
    for i in range(len(users)):
        # Calculate the overlap between the games input by the user and games purchased by each user in the dataset
        coincidence = 0  # Overlap
        positions = []  # Positions of overlapping games in 'games'
        for ii in range(len(games)):
            if games[ii] in np.where(buy[users[i], :] == 1)[0]:
                coincidence += 1
                positions.append(ii)
        if coincidence == 0:
            continue
        simi = []
        for position in positions:
            game = games[position]
            hour = abs(game_hours[position] - hours[users[i], game])
            simi.append(math.exp(-hour))
        similarity[i] = sum(simi) / coincidence
    # Multiply similarity with each row of the player-game matrix
    for i in range(len(users)):
        user = users[i]
        rec[user] = rec[user] * similarity[i]

    new_rec = np.zeros(len(rec[0]))  # 1*n_games matrix
    for i in range(len(new_rec)):
        for user in users:
            new_rec[i] += rec[user][int(i)]
    return new_rec


class Recommendation(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(600, 700)
        self.setWindowTitle('Steam Game Recommendation (for CS777 Team Project 2024)')
        self.setStyleSheet("QWidget { background-color: #333; color: #EEE; font-size: 14px; }"
                           "QPushButton { background-color: #556; border-radius: 5px; }"
                           "QPushButton:hover { background-color: #779; }"
                           "QLabel { font-weight: bold; }"
                           "QComboBox, QLineEdit { background-color: #444; border-radius: 5px; }"
                           "QMessageBox { background-color: #555; }")

        mainLayout = QVBoxLayout(self)

        self.comboBoxes = []
        self.timeEdits = []  # List to hold time labels
        self.buttons = []

        for i in range(5):
            rowLayout = QHBoxLayout()

            label = QLabel(f'Select Your Past Favorite Game {i+1}:')
            comboBox = QComboBox(minimumWidth=200)
            comboBox.setEditable(True)
            self.comboBoxes.append(comboBox)
            timeEdit = QLabel("No Time Set")  # Initially no time is set
            timeEdit.setStyleSheet("color: #AAA;")  # Set lighter color for empty time
            self.timeEdits.append(timeEdit)
            button = QPushButton('Enter Game Time (Hours)')
            self.buttons.append(button)

            rowLayout.addWidget(label)
            rowLayout.addWidget(comboBox)
            rowLayout.addWidget(timeEdit)  # Add the time label to the layout
            rowLayout.addWidget(button)
            mainLayout.addLayout(rowLayout)

        self.bt = QPushButton('Start Recommendation')
        self.bt.clicked.connect(self.recommend)
        mainLayout.addWidget(self.bt)

        self.init_combobox()

        for i, button in enumerate(self.buttons):
            button.clicked.connect(lambda _, x=i: self.timeDialog(x))  # Pass the index x using lambda

    def init_combobox(self):
        for comboBox in self.comboBoxes:
            for game in gamelist:
                comboBox.addItem(game)
            comboBox.setCurrentIndex(-1)
            completer = QCompleter(gamelist, self)
            completer.setFilterMode(Qt.MatchContains)
            completer.setCompletionMode(QCompleter.PopupCompletion)
            comboBox.setCompleter(completer)

    def timeDialog(self, index):
        comboBox = self.comboBoxes[index]
        gameName = comboBox.currentText()
        if gameName:
            gameID = game2idx.get(gameName, None)
            if gameID is not None:
                gamedict[index + 1] = gameName
                idxdict[index + 1] = gameID
                text, ok = QInputDialog.getDouble(self, 'Game Time (Hours)', 'Enter game time (Hours):', min=0.1)
                if ok and text:
                    timedict[index + 1] = text
                    self.timeEdits[index].setText(f"{text} Hours")  # Update the time label
                elif not ok:
                    QMessageBox.information(self, 'Note', 'You canceled the input.', QMessageBox.Close)
            else:
                QMessageBox.information(self, 'Error', 'Please enter a correct game name first!', QMessageBox.Close)
        else:
            QMessageBox.information(self, 'Error', 'Please select a game first!', QMessageBox.Close)

    def recommend(self):
        # Check for unwritten data
        c = 0
        for i in range(1, 6):
            if gamedict[i] == "NULL":
                c += 1
            if idxdict[i] == "NULL":
                c += 1
            if timedict[i] == "NULL":
                c += 1

        # When all data is written
        if c == 0:
            # Show waiting message
            self.waitingMessage = QMessageBox()
            self.waitingMessage.setWindowTitle("Processing")
            self.waitingMessage.setText("Calculating recommendations. Please wait...")
            self.waitingMessage.setStandardButtons(QMessageBox.NoButton)
            self.waitingMessage.open()  # Non-modal dialog, allows processing to continue

            # Convert the dictionary to a list
            usertime = list(timedict.values())
            useridx = list(idxdict.values())

            # Process in the background and then show the result
            QTimer.singleShot(100, lambda: self.processRecommendations(useridx, usertime))  # Adjust the time as needed
        else:
            reply = QMessageBox.information(self, 'Error', 'Please enter all data!', QMessageBox.Close)

    def processRecommendations(self, useridx, usertime):
        # Call the model
        allrecidx = UserSimilarity(useridx, usertime)
        # Sort the data in descending order
        rr = np.argsort(-allrecidx)
        # Get the top five game ids
        top_k = rr[:5]
        recgame.clear()
        for i in top_k:
            recgame.append(idx2game[i])
        # Convert the array to a string and output it
        reclist = "\n".join(recgame)  # Use newline to separate each game name
        # Close the waiting message and show results
        self.waitingMessage.close()
        QMessageBox.information(self, 'Recommended Games', f'Games recommended to you are:\n{reclist}',
                                QMessageBox.Close)


# Main function
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Recommendation()
    w.show()
    # app.exec_() starts the event loop, which returns a status code when the window is closed
    retval = app.exec_()
    # sys.exit(retval) passes the status code to sys.exit, which triggers a SystemExit exception
    # IPython catches this exception and does not throw an error, just prompts how to exit
    sys.exit(retval)

