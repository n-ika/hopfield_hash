# -*- coding: utf-8 -*-

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional 
import torch.utils
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import pandas as pd
import sys

# EP = int(sys.argv[1])
# NUM = int(sys.argv[2])

class AE(torch.nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = torch.nn.Sequential(
                torch.nn.Linear(8, 100),
                torch.nn.ReLU())
        
        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(100,8),
                torch.nn.ReLU())
    
    def forward(self,x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return(encoded,decoded)



"""
Generate training data
"""

def make_data(size=1000):
    onehot = torch.nn.functional.one_hot(torch.arange(0, 8).view(8,1), num_classes=8)

    A = onehot[0]
    C = onehot[2]
    E = onehot[4]
    G = onehot[6]

    B = onehot[1]
    D = onehot[3]
    F = onehot[5]
    H = onehot[7]

    setup = np.arange(4)
    data = setup.repeat(size)
    np.random.shuffle(data)

    data=data[np.insert(np.diff(data).astype(np.bool), 0, True)]


    d = []
    for i,val in enumerate(data):
      if i < data.shape[0]-1:
        if data[i] == 0:
          d.append('AB')
          if data[i+1] == 1:
            d.append('BC')
          elif data[i+1] == 2:
            d.append('BE')
          elif data[i+1] == 3:
            d.append('BG')
        elif data[i] == 1:
          d.append('CD')
          if data[i+1] == 0:
            d.append('DA')
          elif data[i+1] == 2:
            d.append('DE')
          elif data[i+1] == 3:
            d.append('DG')
        elif data[i] == 2:
          d.append('EF')
          if data[i+1] == 0:
            d.append('FA')
          elif data[i+1] == 1:
            d.append('FC')
          elif data[i+1] == 3:
            d.append('FG')  
        elif data[i] == 3:
          d.append('GH')
          if data[i+1] == 0:
            d.append('HA')
          elif data[i+1] == 1:
            d.append('HC')
          elif data[i+1] == 2:
            d.append('HE')  

    d = np.array(d)


    stims = np.ones((d.shape[0],A.shape[1]))*-1
    stims[np.where(d=='AB')[0]] = A*0.9+B
    stims[np.where(d=='BE')[0]] = B*0.9+E
    stims[np.where(d=='BG')[0]] = B*0.9+G
    stims[np.where(d=='BC')[0]] = B*0.9+C
    stims[np.where(d=='CD')[0]] = C*0.9+D
    stims[np.where(d=='DE')[0]] = D*0.9+E
    stims[np.where(d=='DA')[0]] = D*0.9+A
    stims[np.where(d=='DG')[0]] = D*0.9+G
    stims[np.where(d=='EF')[0]] = E*0.9+F
    stims[np.where(d=='FA')[0]] = F*0.9+A
    stims[np.where(d=='FC')[0]] = F*0.9+C
    stims[np.where(d=='FG')[0]] = F*0.9+G
    stims[np.where(d=='GH')[0]] = G*0.9+H
    stims[np.where(d=='HA')[0]] = H*0.9+A
    stims[np.where(d=='HC')[0]] = H*0.9+C
    stims[np.where(d=='HE')[0]] = H*0.9+E

    return(torch.Tensor(stims),onehot)


def train_model(tr_data, epochs, NUM):
    """
    Train the model
    """

    model = AE()
    loss_fun = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    loss_track = []
    # Train
    for epoch in range(epochs):
        # Forward
        codes, decoded = model(tr_data)

        # Backward
        optimizer.zero_grad()
        loss = loss_fun(decoded, tr_data)
        loss_track.append(float(loss))
        loss.backward()
        optimizer.step()

        # Show progress
        print('[{}/{}] Loss:'.format(epoch+1, epochs), loss.item())


    # Save
    df_loss = pd.DataFrame(loss_track)
    df_loss.to_csv('./models/loss/loss_'+'_'.join([str(epochs),str(NUM)])+'.csv')
    torch.save(model.state_dict(), './models/ae_'+'_'.join([str(epochs),str(NUM)])+'.pth')

    return(model)


"""
Track the loss
"""

# plt.plot(loss_track,label=str(NUM))
# plt.legend()
# plt.savefig("./figs/.jpg")

def run_experiment(tr_data, onehot, EP, NUM):
    """
    Get reconstruction results
    """

    model = train_model(tr_data, epochs=EP, NUM=NUM)
    As = onehot[0::2]
    Bs = onehot[1::2]

    results = {"idx_a" : [],"idx_b" : [],"a2a" : [],"b2b" : [],"a2b" : [],"b2a" : []}

    rec = model(As.float())[1]
    r = rec.detach().numpy()
    r = np.reshape(r, (4,8))

    A = As.detach().numpy()
    A = np.reshape(A, (4,8))

    for i in range(A.shape[0]):
      id_aa = np.where(A[i]==1)[0]
      id_ab = id_aa+1
      results["a2a"].append(float(r[i][id_aa]))
      results["a2b"].append(float(r[i][id_ab]))
      results["idx_a"].append(float(int(id_aa)))


    rec = model(Bs.float())[1]
    r = rec.detach().numpy()
    r = np.reshape(r, (4,8))

    A = As.detach().numpy()
    A = np.reshape(A, (4,8))

    B = Bs.detach().numpy()
    B = np.reshape(B, (4,8))
    for i in range(B.shape[0]):
      id_bb = np.where(B[i]==1)[0]
      id_ba = id_bb-1
      results["b2b"].append(float(r[i][id_bb]))
      results["b2a"].append(float(r[i][id_ba]))
      results["idx_b"].append(float(int(id_bb)))

    res=pd.DataFrame(results)
    res.to_csv("./results/res_"+"_".join([str(EP),str(NUM)])+".csv")

    return()