import os
import traceback
import wmi
from PIL import Image

import cv2, numpy as np
from datetime import datetime
from flask import Flask, render_template, url_for, redirect, request, Response
from waitress import serve
from main import main

diretorio = os.getcwd().split('\\')[-1]
app = Flask(__name__, static_folder=f"../{diretorio}", template_folder=f"../{diretorio}")

@app.route('/', methods=["POST", "GET"])
def index():
    nomeFotoArmazenada = 'fotos/foto.png'
    if request.method == 'POST':
        fotoCarregada = request.files['fotoCarregada']
        if fotoCarregada.filename != '':
            fotoCarregada.save(nomeFotoArmazenada)
            fotoCarregada.save(f'fotos/{fotoCarregada.filename}')
            print(fotoCarregada)
    if os.path.isfile(nomeFotoArmazenada):
        arquivoExite = True
        try:
            img = Image.open(nomeFotoArmazenada)
            width = img.width
            height = img.height
        except:
          width = ''
          height = ''
    else:
        width = ''
        height = ''
        arquivoExite = False
    return render_template('index.html', arquivoExite = arquivoExite, width = width, height = height)

@app.route('/removerFoto', methods=["POST", "GET"])
def removFoto():
    nomeFotoArmazenada = 'fotos/foto.png'
    if os.path.isfile(nomeFotoArmazenada):
        os.remove(nomeFotoArmazenada)
        return redirect('/')

if __name__ == '__main__':
    # serve(app, host='0.0.0.0', port=5000)
    # Import module


    # Initializing the wmi constructor
    f = wmi.WMI()

    flag = 0

    # Iterating through all the running processes
    for process in f.Win32_Process():
      if "tccApp.exe" == process.Name:
        print("Application is Running")
        flag = 1
        break

    if flag == 0:
      print("Application is not Running")
    app.run(host='0.0.0.0', debug=True, port=5000)
