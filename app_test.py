import webbrowser
import PySimpleGUI as sg
sg.theme('DarkRed2')   # Add a little color to your windows
# All the stuff inside your window. This is the PSG magic code compactor...
layout = [  [sg.Text(auto_size_text=True, 'Welcome to TrashStop')],
            [sg.Text('Enter something on Row 2'), sg.InputText()]]

# Create the Window
window = sg.Window('Window Title', layout, no_titlebar=True).Finalize()
window.Maximize()


# Event Loop to process "events"
while True:             
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Cancel'):
        break

window.close()

'''
while True:
    nb = int(input('Choose a number: '))

    if nb==0:
        #nothing
        webbrowser.open('http://127.0.0.1:5000/')

    elif nb==1:
        #landfill
        webbrowser.open('http://127.0.0.1:5000/landfill')

'''