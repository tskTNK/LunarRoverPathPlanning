import matplotlib.pyplot as plt
import numpy as np
import matlab.engine
import csv
import math
import os
import statistics
from numpy.linalg import multi_dot
from ai import cs
import rasterio as rio
from rasterio.plot import show
from gym_minigrid.simulation_param import *


## Creat slope map from DEM by python

# read DEM file
dem = rio.open("2500-2500.tif")
dem_array = dem.read(1).astype('float64')

np.savetxt('DEM.csv',dem_array, fmt = '%d', delimiter=",")

# plt.figure()
# plt.imshow(dem_array)
# plt.colorbar()
# plt.show()

explorationDEM = np.zeros((mapsize[0], mapsize[1]))
for a in range(mapsize[0]):
    for b in range(mapsize[1]):
        explorationDEM[b][a] = dem_array[location[0]+a][location[1]+b]

xlength = explorationDEM.shape[0]
ylength = explorationDEM.shape[1]

# create slope map
minAlt = np.min(explorationDEM)
explorationDEM = explorationDEM - minAlt*np.ones((xlength, ylength))
explorationDEM = explorationDEM/slopeScaler

slope = np.zeros((xlength, ylength))
slope2dAngle = np.zeros((xlength, ylength, 2))
slopeSurfaceNormal = np.zeros((xlength, ylength, 3)) 

for i in range(xlength):
    for j in range(ylength):

        print(i, j, explorationDEM[j][i])

        if i != xlength-1 and j != ylength-1:
            xdiff = explorationDEM[i][j+1]-explorationDEM[i][j]
            ydiff = explorationDEM[i+1][j]-explorationDEM[i][j]
            vectorXdirection = [map_reso, 0, xdiff]
            vectorYdirection = [0, map_reso, ydiff]
            slope2dAngle[i][j][0] = np.rad2deg(math.atan2(vectorXdirection[2],vectorXdirection[0])) # rotation angle around Y axis
            slope2dAngle[i][j][1] = np.rad2deg(math.atan2(vectorYdirection[2],vectorYdirection[1])) # rotation angle around X axis
            surfaceNormal = np.cross(vectorXdirection,vectorYdirection)
            verticalNormal = [0, 0, 1]
            Theta = np.rad2deg(math.acos(np.dot(surfaceNormal,verticalNormal)/(np.linalg.norm(surfaceNormal)*np.linalg.norm(verticalNormal))))
            slope[i][j]=Theta
            slopeSurfaceNormal[i][j][0] = surfaceNormal[0]
            slopeSurfaceNormal[i][j][1] = surfaceNormal[1]
            slopeSurfaceNormal[i][j][2] = surfaceNormal[2]

        if i == xlength-1:
            slope[i][j] = slope[i-1][j]
            slope2dAngle[i][j][:] = slope2dAngle[i-1][j][:]
            slopeSurfaceNormal[i][j][:] = slopeSurfaceNormal[i-1][j][:]

        if j == ylength-1:
            slope[i][j] = slope[i][j-1]
            slope2dAngle[i][j][:] = slope2dAngle[i][j-1][:]
            slopeSurfaceNormal[i][j][:] = slopeSurfaceNormal[i][j-1][:]

# plt.figure()
# plt.imshow(slope)
# plt.colorbar()
# plt.show()

# slopeAngle
filename = "slope.csv"
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(slope)

# slope2dAngle
test = np.zeros((xlength, ylength, 2))
for a in range(xlength):
    for b in range(ylength):
        for c in range(2):
            test[a][b][c] = slope2dAngle[a][b][c]
slope2dAngle_reshaped = test.reshape(test.shape[0], -1) # reshaping the array from 3D matrice to 2D matrice.
filename = "slope2dAngle.csv"
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(slope2dAngle_reshaped)

# slopeSurfaceNormal
test = np.zeros((xlength, ylength, 3))
for a in range(xlength):
    for b in range(ylength):
        for c in range(3):
            test[a][b][c] = slopeSurfaceNormal[a][b][c]
slopeSurfaceNormal_reshaped = test.reshape(test.shape[0], -1) # reshaping the array from 3D matrice to 2D matrice.
filename = "slopeSurfaceNormal.csv"
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(slopeSurfaceNormal_reshaped)


## Calculate lunar surface environmental data

# sun vector calculation
xSun = np.zeros(max_env_steps_for_graphmaking)
ySun = np.zeros(max_env_steps_for_graphmaking)
zSun = np.zeros(max_env_steps_for_graphmaking)
Vsun = np.zeros((max_env_steps_for_graphmaking,3))

for t in range(max_env_steps_for_graphmaking):
    sunCurrentTheta = sunInitialTheta
    sunCurrentPhi = sunInitialPhi - 360*t*timePerStep/(29.5*24*60)
    # spherical to cartesian
    xSun[t], ySun[t], zSun[t] = cs.sp2cart(r=1, theta=np.deg2rad(sunInitialTheta), phi=np.deg2rad(sunCurrentPhi)) # r, theta, phi
    Vsun[t] = [xSun[t], ySun[t], zSun[t]]

# Set Surface Normal for each grid
slopeSurfaceNormalVecs = np.zeros((xlength,ylength,3))

for i in range(xlength):
    for j in range(ylength):
        slopeSurfaceNormalVecs[i][j] = slopeSurfaceNormal[i][j][0]*TopoX+slopeSurfaceNormal[i][j][1]*TopoY+slopeSurfaceNormal[i][j][2]*TopoZ
        A = slopeSurfaceNormalVecs[i][j]
        slopeSurfaceNormalVecs[i][j] = slopeSurfaceNormalVecs[i][j]/np.linalg.norm(A)

# Calculate Temporal Sun angle at each grid
slopeSurfaceNormalVecs_temporary= np.zeros(3)
sunAngles = np.zeros((xlength,ylength,max_env_steps_for_graphmaking))

# this calculation is very slow when it's executed in minigrid.py
for i in range(xlength):
    for j in range(ylength):
        for t in range(max_env_steps_for_graphmaking):
            slopeSurfaceNormalVecs_temporary= slopeSurfaceNormalVecs[i][j]
            sunAngles_temporary = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(slopeSurfaceNormalVecs_temporary, Vsun[t])), np.dot(slopeSurfaceNormalVecs_temporary, Vsun[t])))
            sunAngles[i][j][t] = min(sunAngles_temporary, 90)
            if i % 10 == 0 and j % 10 == 0 and t % 10 == 0:
                print("progress", i, j, t)

# print(sunAngles[0][0]) # confirmed that the values are same as matlab
lunarSurfaceTemp = np.zeros((xlength,ylength,max_env_steps_for_graphmaking))

for i in range(xlength):
    for j in range(ylength):
        for t in range(max_env_steps_for_graphmaking):
            if sunAngles[i][j][t] == 90:
                lunarSurfaceTemp[i][j][t] = 30 # user-set fixed surface tempearture in shadowed regions
            else:
                lunarSurfaceTemp[i][j][t] = np.power((absorptivitySurface * np.cos(np.deg2rad(sunAngles[i][j][t])) * P_s)/(emissivitySurface * stephanBoltz), 0.25)
            lunarSurfaceTemp[i][j][t] = lunarSurfaceTemp[i][j][t] - 273.15
            if i % 10 == 0 and j % 10 == 0 and t % 10 == 0:
                print("progress2", i, j, t)

# reshaping the array from 3D matrice to 2D matrice.
lunarSurfaceTemp_reshaped = lunarSurfaceTemp.reshape(lunarSurfaceTemp.shape[0], -1)
filename = "lunarSurfaceTemp.csv"
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(lunarSurfaceTemp_reshaped)

# reshaping the array from 3D matrice to 2D matrice.
slopeSurfaceNormalVecs_reshaped = slopeSurfaceNormalVecs.reshape(slopeSurfaceNormalVecs.shape[0], -1)
filename = "slopeSurfaceNormalVecs.csv"
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(slopeSurfaceNormalVecs_reshaped)

filename = "Vsun.csv"
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(Vsun)

# reshaping the array from 3D matrice to 2D matrice.
sunAngles_to_the_moon_reshaped = sunAngles.reshape(sunAngles.shape[0], -1)
filename = "sunAngles_to_the_moon.csv"
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(sunAngles_to_the_moon_reshaped)


## Test rover temperature calculation and TP cost function

# # Rover type
# RoverType = 2 # 1: well isolated heat connection, 2: moderate heat connection
# nodelength = 8
# panelLength = 6

# # Set Node IDs
# top = 0
# side1 = 1
# side2 = 2
# front = 3
# rear = 4
# bottom = 5
# outerspace = 6
# lunarsurface = 7

# acceleratingFactor = 240
# epochcut = 1
# roverOrientationResolution = 30
# roverOrientationAll = np.arange(0, 180, roverOrientationResolution, dtype=int)
# missionDaysNo = 10;
# missionDuration = missionDaysNo*24*60;  # minutes basis
# interval = 30; # [minutes] -> it takes 'interval' minutes to move one grid
# timeVector = np.arange(0, missionDuration, interval, dtype=int)
# timevec = np.arange(0, missionDuration/interval+1, dtype=int) # 0-384

# # power calc param set
# solarPanelEff = 0.28
# solarPanelRatio2surfaceAera = 0.50
# solarPowerConversionEff = 0.80
# powerConversionLoss = 0.80

# iterationTime = int(12000/acceleratingFactor)
# iterationTime2 = int(interval*60/acceleratingFactor/epochcut)

# # read slope data from csv
# CSVData3 = open("lunarSurfaceTemp.csv")
# lunarSurfaceTemp = np.loadtxt(CSVData3, delimiter=",")
# lunarSurfaceTemp = lunarSurfaceTemp.reshape(xlength,ylength,max_env_steps_for_graphmaking)

# # read slope data from csv
# CSVData4 = open("slopeSurfaceNormalVecs.csv")
# slopeSurfaceNormalVecs = np.loadtxt(CSVData4, delimiter=",")
# slopeSurfaceNormalVecs = slopeSurfaceNormalVecs.reshape(xlength,ylength,3)

# # read slope data from csv
# CSVData5 = open("Vsun.csv")
# Vsun = np.loadtxt(CSVData5, delimiter=",")

# # read slope data from csv
# CSVData6 = open("slope2dAngle.csv")
# slope2dAngle_ = np.loadtxt(CSVData6, delimiter=",")
# slope2dAngle_ = slope2dAngle_.reshape(xlength,ylength,2)

# class Node:

#     def __init__(self, temp=0, radiationIn=0, conductionIn=0, heatInput=0, \
#                 heatCapacitance=0, typeConstant=0, emissivity=0, absorptivity=0, \
#                 spheat=0, density=0, surface=0, volume=0, mass=0):
#         self.temp = temp
#         self.radiationIn = radiationIn
#         self.conductionIn = conductionIn
#         self.heatInput = heatInput
#         self.heatCapacitance = heatCapacitance
#         self.typeConstant = typeConstant
#         self.emissivity = emissivity
#         self.absorptivity = absorptivity
#         self.spheat = spheat
#         self.density = density
#         self.surface = surface
#         self.volume = volume
#         self.mass = mass

# class Connection:

#     def __init__(self, emissivity1=1, emissivity2=1, radSurface=0, viewFactor=0, \
#                 thermalContactResistance=0, thermalContactArea=0, innerThermalConductivity=0, \
#                 innerConductiveArea=0, innerConductiveLength=1, temp1=0, temp2=0):
#         self.emissivity1 = emissivity1
#         self.emissivity2 = emissivity2
#         self.radSurface = radSurface
#         self.viewFactor = viewFactor
#         self.thermalContactResistance = thermalContactResistance
#         self.thermalContactArea = thermalContactArea
#         self.innerThermalConductivity = innerThermalConductivity
#         self.innerConductiveArea = innerConductiveArea
#         self.innerConductiveLength = innerConductiveLength
#         self.temp1 = temp1
#         self.temp2 = temp2

#         stephanBoltz = 5.67e-8
#         # qrad = -1*stephanBoltz*radSurface*viewFactor*emissivity1*emissivity2*(np.power(temp1, 4)-np.power(temp2, 4))
#         qrad = -1*stephanBoltz*radSurface*viewFactor*(1/(1/emissivity1+1/emissivity2-1))*(np.power(temp1, 4)-np.power(temp2, 4)) # surface facing in parallel

#         self.Qrad1 = qrad
#         self.Qrad2 = -1*qrad

#         cond = -1*thermalContactResistance*thermalContactArea*(temp1-temp2)
#         cond2 = -1*innerThermalConductivity*innerConductiveArea/innerConductiveLength*(temp1-temp2)

#         self.Qcond1 = cond+cond2
#         self.Qcond2 = -1*(cond+cond2)

# def map(nodeId1, nodeId2, nodeLength):
#     return nodeId1*nodeLength + nodeId2

# class Panel:

#     def __init__(self, absorptivity=0, emissivity=0, surfaceArea=0, \
#                 thickness=0, spheat=0, density=0, viewFactor2moon=0, \
#                 elecDissip=0, theta=0, phi=0):
#         self.absorptivity = absorptivity
#         self.emissivity = emissivity
#         self.surfaceArea = surfaceArea
#         self.thickness = thickness
#         self.spheat = spheat
#         self.density = density
#         self.viewFactor2space = 1-viewFactor2moon
#         self.viewFactor2moon = viewFactor2moon
#         self.elecDissip = elecDissip
#         self.theta = theta
#         self.phi = phi

#     def normalVec(self, roverOrientation, X, Y, Z):
#         vn = np.multiply(X, np.cos(np.deg2rad(roverOrientation + self.phi))) + np.multiply(Y, np.sin(np.deg2rad(roverOrientation + self.phi)))
#         vn = np.multiply(vn, np.cos(np.deg2rad(self.theta))) + np.multiply(Z, np.sin(np.deg2rad(self.theta)))
#         return vn

# panels = [Panel() for i in range(panelLength)]

# # Top panel
# topSurface = 0.50 * 0.25 # length[m], width[m]
# electronicsPower = 4.5
# panels[top] = Panel(0.08, 0.95, topSurface, 0.01, 1.04, 1.8, 0, electronicsPower, 90, 0)

# # Side 1 panel
# sideSurface1 = 0.5 * 0.2 # length[m], width[m]
# panels[side1] = Panel(0.75, 0.81, sideSurface1, 0.01, 1.04, 1.8, 0.35, 0, 20, 90)

# # Side 2 panel
# sideSurface2 = 0.5 * 0.2 # length[m], width[m]
# panels[side2] = Panel(0.75, 0.81, sideSurface2, 0.01, 1.04, 1.8, 0.35, 0, 20, 270)

# # Front panel
# frontSurface = 0.25 * 0.2 # length[m], width[m]
# panels[front] = Panel(0.08, 0.95, frontSurface, 0.01, 1.04, 1.8, 0.39, 0, 15, 0)

# # Rear panel
# rearSurface = 0.25 * 0.2 # length[m], width[m]
# panels[rear] = Panel(0.08, 0.95, rearSurface, 0.01, 1.04, 1.8, 0.39, 0, 15, 180)

# # Bottom panel
# bottomSurface = 0.6 * 0.30 # length[m], width[m]
# panels[bottom] = Panel(0.05, 0.05, bottomSurface, 0.01, 1.04, 1.8, 1, 0, -90, 0)

# # Inner View Factor
#                 #    TOP  SIDE1  SIDE2  FRONT  REAR  BOTTOM
# innerViewFactor = [ [0, 0.1, 0.1, 0.1, 0.1, 0.5],  # TOP
#                     [0.1, 0, 0.5, 0.1, 0.1, 0.1],  # SIDE1
#                     [0.1, 0.5, 0, 0.1, 0.1, 0.1],  # SIDE2
#                     [0.1, 0.1, 0.1, 0, 0.5, 0.1],  # FRONT
#                     [0.1, 0.1, 0.1, 0.5, 0, 0.1],  # REAR
#                     [0.5, 0.1, 0.1, 0.1, 0.1, 0]] # BOTTOM

# nodes = [Node() for i in range(nodelength)]
# connections = [Connection() for i in range(nodelength*nodelength)]

# # Thermal Contact Resistance [W/m2/K]
# # 10000: metal to metal connection, 1000: standard isolation, 100: high isolation
# if RoverType  == 1:
#     thermalCR_SIDE2TOP = 1
#     thermalCR_BOTTOM2SIDE = 1
#     thermalCR_BOTTOM2LunarSurface = 1
# elif RoverType == 2:
#     thermalCR_SIDE2TOP = 1800
#     thermalCR_BOTTOM2SIDE = 1800
#     thermalCR_BOTTOM2LunarSurface = 300

# # Thermal Contact Surface [m2]
# thermalCS_SIDE2TOP = 0.01*0.1
# thermalCS_FR2TOP = 0.01*0.05
# thermalCS_BOTTOM2SIDE = 0.01*0.1
# thermalCS_BOTTOM2LunarSurface = 0.05*0.05*4

# # Emissivity
# nodes[top].emissivity= panels[top].emissivity
# nodes[side1].emissivity = panels[side1].emissivity
# nodes[side2].emissivity = panels[side2].emissivity
# nodes[front].emissivity = panels[front].emissivity
# nodes[rear].emissivity = panels[rear].emissivity
# nodes[bottom].emissivity = panels[bottom].emissivity
# nodes[outerspace].emissivity = 1.00
# nodes[lunarsurface].emissivity = emissivitySurface

# # Absorptivity
# nodes[top].absorptivity= panels[top].absorptivity
# nodes[side1].absorptivity = panels[side1].absorptivity
# nodes[side2].absorptivity = panels[side2].absorptivity
# nodes[front].absorptivity = panels[front].absorptivity
# nodes[rear].absorptivity = panels[rear].absorptivity
# nodes[bottom].absorptivity = panels[bottom].absorptivity

# # Specific heat [Ws/gK]
# nodes[top].spheat = panels[top].spheat
# nodes[side1].spheat = panels[side1].spheat
# nodes[side2].spheat = panels[side2].spheat
# nodes[front].spheat = panels[front].spheat
# nodes[rear].spheat = panels[rear].spheat
# nodes[bottom].spheat = panels[bottom].spheat

# # Density [g/cm3]
# nodes[top].density = panels[top].density
# nodes[side1].density = panels[side1].density
# nodes[side2].density = panels[side2].density
# nodes[front].density = panels[front].density
# nodes[rear].density = panels[rear].density
# nodes[bottom].density = panels[bottom].density

# # Volume [cm3]
# nodes[top].volume = panels[top].surfaceArea * panels[top].thickness *100*100*100
# nodes[side1].volume = panels[side1].surfaceArea * panels[side1].thickness *100*100*100
# nodes[side2].volume = panels[side2].surfaceArea * panels[side2].thickness *100*100*100
# nodes[front].volume = panels[front].surfaceArea * panels[front].thickness *100*100*100
# nodes[rear].volume = panels[rear].surfaceArea * panels[rear].thickness *100*100*100
# nodes[bottom].volume = panels[bottom].surfaceArea * panels[bottom].thickness *100*100*100

# # surface [m2]
# nodes[top].surface = panels[top].surfaceArea
# nodes[side1].surface = panels[side1].surfaceArea
# nodes[side2].surface = panels[side2].surfaceArea
# nodes[front].surface = panels[front].surfaceArea
# nodes[rear].surface = panels[rear].surfaceArea
# nodes[bottom].surface = panels[bottom].surfaceArea

# # Mass of elements
# for i in range(nodelength):
#     nodes[i].mass = nodes[i].density*nodes[i].volume # [g]
#     # print("volume", nodes[i].volume, "mass", nodes[i].mass, "spheat", nodes[i].spheat)

# # total mass
# totalmass = 0
# for i in range(nodelength):
#     totalmass = totalmass + nodes[i].mass

# # heat Capacitanceacitance
# for i in range(panelLength):
#     nodes[i].heatCapacitance = nodes[i].spheat*nodes[i].mass # [Ws/K]
#     # print("capacitance", nodes[i].heatCapacitance)

# nodes[outerspace].heatCapacitance = 1000000
# nodes[lunarsurface].heatCapacitance = 1000000

# # define constant-temperature nodes
# nodes[outerspace].typeConstant = 1
# nodes[lunarsurface].typeConstant = 1

# # Set radiative heat property and conductive heat property
# connections[map(top,outerspace,nodelength)].radSurface = panels[top].surfaceArea
# connections[map(top,outerspace,nodelength)].viewFactor = panels[top].viewFactor2space
# connections[map(side1,outerspace,nodelength)].radSurface = panels[side1].surfaceArea
# connections[map(side1,outerspace,nodelength)].viewFactor = panels[side1].viewFactor2space
# connections[map(side2,outerspace,nodelength)].radSurface = panels[side2].surfaceArea
# connections[map(side2,outerspace,nodelength)].viewFactor = panels[side2].viewFactor2space
# connections[map(front,outerspace,nodelength)].radSurface = panels[front].surfaceArea
# connections[map(front,outerspace,nodelength)].viewFactor = panels[front].viewFactor2space
# connections[map(rear,outerspace,nodelength)].radSurface = panels[rear].surfaceArea
# connections[map(rear,outerspace,nodelength)].viewFactor = panels[rear].viewFactor2space
# connections[map(bottom,outerspace,nodelength)].radSurface = panels[bottom].surfaceArea
# connections[map(bottom,outerspace,nodelength)].viewFactor = panels[bottom].viewFactor2space

# connections[map(top,lunarsurface,nodelength)].radSurface = panels[top].surfaceArea
# connections[map(top,lunarsurface,nodelength)].viewFactor = panels[top].viewFactor2moon
# connections[map(side1,lunarsurface,nodelength)].radSurface = panels[side1].surfaceArea
# connections[map(side1,lunarsurface,nodelength)].viewFactor = panels[side1].viewFactor2moon
# connections[map(side2,lunarsurface,nodelength)].radSurface = panels[side2].surfaceArea
# connections[map(side2,lunarsurface,nodelength)].viewFactor = panels[side2].viewFactor2moon
# connections[map(front,lunarsurface,nodelength)].radSurface = panels[front].surfaceArea
# connections[map(front,lunarsurface,nodelength)].viewFactor = panels[front].viewFactor2moon
# connections[map(rear,lunarsurface,nodelength)].radSurface = panels[rear].surfaceArea
# connections[map(rear,lunarsurface,nodelength)].viewFactor = panels[rear].viewFactor2moon
# connections[map(bottom,lunarsurface,nodelength)].radSurface = panels[bottom].surfaceArea
# connections[map(bottom,lunarsurface,nodelength)].viewFactor = panels[bottom].viewFactor2moon

# for ID1 in range(panelLength): # 0-5
#     for ID2 in range(ID1, panelLength):
#         # print("ID1, ID2", ID1, ID2)
#         connections[map(ID1,ID2,nodelength)].radSurface = panels[ID1].surfaceArea
#         connections[map(ID1,ID2,nodelength)].viewFactor = innerViewFactor[ID1][ID2]
#         # if ID2 == 5:
#         #     print(innerViewFactor[ID1][ID2])

# connections[map(top,side1,nodelength)].thermalContactArea = thermalCS_SIDE2TOP
# connections[map(top,side1,nodelength)].thermalContactResistance = thermalCR_SIDE2TOP
# connections[map(top,side2,nodelength)].thermalContactArea = thermalCS_SIDE2TOP
# connections[map(top,side2,nodelength)].thermalContactResistance = thermalCR_SIDE2TOP
# connections[map(top,front,nodelength)].thermalContactArea = thermalCS_FR2TOP
# connections[map(top,front,nodelength)].thermalContactResistance = thermalCR_SIDE2TOP
# connections[map(top,rear,nodelength)].thermalContactArea = thermalCS_FR2TOP
# connections[map(top,rear,nodelength)].thermalContactResistance = thermalCR_SIDE2TOP
# connections[map(bottom,side1,nodelength)].thermalContactArea = thermalCS_BOTTOM2SIDE
# connections[map(bottom,side1,nodelength)].thermalContactResistance = thermalCR_BOTTOM2SIDE
# connections[map(bottom,side2,nodelength)].thermalContactArea = thermalCS_BOTTOM2SIDE
# connections[map(bottom,side2,nodelength)].thermalContactResistance = thermalCR_BOTTOM2SIDE
# connections[map(bottom,front,nodelength)].thermalContactArea = thermalCS_BOTTOM2SIDE
# connections[map(bottom,front,nodelength)].thermalContactResistance = thermalCR_BOTTOM2SIDE
# connections[map(bottom,rear,nodelength)].thermalContactArea = thermalCS_BOTTOM2SIDE
# connections[map(bottom,rear,nodelength)].thermalContactResistance = thermalCR_BOTTOM2SIDE

# connections[map(bottom,lunarsurface,nodelength)].thermalContactArea = thermalCS_BOTTOM2LunarSurface
# connections[map(bottom,lunarsurface,nodelength)].thermalContactResistance = thermalCR_BOTTOM2LunarSurface

# for ID in range(panelLength):
#     nodes[ID].heatCapacitance = nodes[ID].heatCapacitance/acceleratingFactor
#     # print("heatCapacitance", nodes[ID].heatCapacitance)


# # variable for calculation
# nodeTemp = np.zeros((nodelength, iterationTime))
# nodeQtotal = np.zeros((nodelength, iterationTime))
# nodeHeatInput = np.zeros((nodelength, iterationTime))
# nodeRadInput = np.zeros((nodelength, iterationTime))
# nodeCondInput = np.zeros((nodelength, iterationTime))
# roverAllTemp = np.zeros((panelLength, len(timeVector), len(roverOrientationAll)))
# panelNormalVecs = np.zeros((3, panelLength))
# sunAnglesToPanels = np.zeros((panelLength))
# Q_in_sun = np.zeros((panelLength))
# Q_in_la = np.zeros((panelLength))
# Q_in_e = np.zeros((panelLength))
# solarPowerDischarge = np.zeros((panelLength))
# Q_total = np.zeros((panelLength))
# MonitorQrad = np.zeros((nodelength, nodelength))
# MonitorQcond = np.zeros((nodelength, nodelength))
# powerGen = np.zeros((len(timeVector), len(roverOrientationAll)))

# agent_pos = [5,4] # temporary position
# step_count = 1

# # set environmental temp
# nodes[outerspace].temp = 3;
# nodes[lunarsurface].temp = lunarSurfaceTemp[agent_pos[0],agent_pos[1],step_count] + 273.15; # degC -> K

# # creat Body vector at each epoch with regard to the cartesian frame
# rotAngleAroundTopoY = slope2dAngle_[agent_pos[0],agent_pos[1],0]
# rotAngleAroundTopoX = slope2dAngle_[agent_pos[0],agent_pos[1],1]
# # print("inclination", rotAngleAroundTopoX, rotAngleAroundTopoY)
# # rotate around the current Y axis by localLongitude
# rotationMat1LS =  [[np.cos(np.deg2rad(rotAngleAroundTopoY)), 0, np.sin(np.deg2rad(rotAngleAroundTopoY))],
#                     [0, 1, 0],
#                     [-np.sin(np.deg2rad(rotAngleAroundTopoY)), 0, np.cos(np.deg2rad(rotAngleAroundTopoY))]]
# # rotate around the current X axis by localLattitude
# rotationMat2LS =  [[1, 0, 0],
#                     [0, np.cos(np.deg2rad(rotAngleAroundTopoX)), -np.sin(np.deg2rad(rotAngleAroundTopoX))],
#                     [0, np.sin(np.deg2rad(rotAngleAroundTopoX)), np.cos(np.deg2rad(rotAngleAroundTopoX))]]
# # rotate frame with regard to the current frame
# rotationLS = np.dot(rotationMat1LS, rotationMat2LS)

# # print(rotationLS)

# SurfaceX = np.dot(TopoX, rotationLS)
# SurfaceY = np.dot(TopoY, rotationLS)
# SurfaceZ = np.dot(TopoZ, rotationLS)

# for pose in range(len(roverOrientationAll)):

#     # print("pose", pose)

#     t = step_count

#     # initialization
#     for i in range(nodelength):
#         nodes[i].radiationIn = 0
#         nodes[i].conductionIn = 0
#         nodes[i].heatInput = 0

#     # Set temoporary rover orientation
#     roverOrientationTemp = roverOrientationAll[pose]

#     # Calculate Sun Angle relative to panels
#     for i in range(panelLength):
#         # create top vectors based on the current rover orientation and body vector
#         panelNormalVecs[:,i] = panels[i].normalVec(roverOrientation=roverOrientationTemp, X=SurfaceX, Y=SurfaceY, Z=SurfaceZ)
#         # cap the sun angle value less than 90 using min fucntion
#         sunAngles_temporary = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(panelNormalVecs[:,i], Vsun[t])), np.dot(panelNormalVecs[:,i], Vsun[t])))
#         sunAnglesToPanels[i] = min(sunAngles_temporary, 90)

#     # print("Vsun", Vsun[t])

#     # print("normalVecs")
#     # for i in range(panelLength):
#     #     print(normalVecs[:,i])

#     # print("sunAnglesToPanels")
#     # for i in range(panelLength):
#     #     print(sunAnglesToPanels[i])

#     # Calculate Solar radiation
#     for i in range(panelLength):
#         Q_in_sun[i] = panels[i].absorptivity * np.cos(np.deg2rad(sunAnglesToPanels[i])) * panels[i].surfaceArea * P_s

#     # print("Q_in_sun")
#     # for i in range(panelLength):
#     #     print(Q_in_sun[i])

#     # Detect shadow flag
#     shadowFlag = 0
#     if all(item == 90 for item in sunAnglesToPanels):
#         shadowFlag = 1

#     # Calculate Lunar albedo
#     for i in range(panelLength):
#         Q_in_la[i] = panels[i].absorptivity * panels[i].viewFactor2moon * panels[i].surfaceArea * (1-absorptivitySurface) * P_s

#     # Calculate Lunar radiation
#     Pe = emissivitySurface*stephanBoltz*np.power(nodes[lunarsurface].temp, 4);
#     for i in range(panelLength):
#         Q_in_e[i] = panels[i].emissivity * panels[i].viewFactor2moon * panels[i].surfaceArea * Pe

#     # Calculate Elec Power
#     electronicsPowerSum = 0;
#     if shadowFlag == 0:
#         for i in range(panelLength):
#             electronicsPowerSum = electronicsPowerSum + panels[i].elecDissip

#     # Solar power discharging
#     if Q_in_sun[side1]*solarPanelEff*solarPanelRatio2surfaceAera- electronicsPowerSum > Q_in_sun[side2]*solarPanelEff*solarPanelRatio2surfaceAera:
#         solarPowerDischarge[side1] = -electronicsPowerSum
#     elif Q_in_sun[side2]*solarPanelEff*solarPanelRatio2surfaceAera- electronicsPowerSum > Q_in_sun[side1]*solarPanelEff*solarPanelRatio2surfaceAera:
#         solarPowerDischarge[side2] = -electronicsPowerSum
#     else:
#         solarPowerDischarge[side1] = -electronicsPowerSum/2
#         solarPowerDischarge[side2] = -electronicsPowerSum/2

#     # Set initial temperature for multi epoch calculation
#     if t == 1:
#         for i in range(panelLength):
#             Q_total[i] = Q_in_sun[i] + Q_in_la[i] + Q_in_e[i] + panels[i].elecDissip + solarPowerDischarge[i]
#             nodes[i].temp = np.power(Q_total[i]/(panels[i].emissivity * panels[i].surfaceArea * stephanBoltz), 0.25)
#     else:
#         # carry over temperatures from the last epoch
#         for i in range(panelLength):
#             nodes[i].temp = roverAllTemp[i][t-1][pose]

#     # print("Q_total")
#     # for i in range(panelLength):
#     #     print(Q_total[i])

#     # print("Node_temp")
#     # for i in range(panelLength):
#     #     print(nodes[i].temp)

#     # Thermal calculation with inner conduction and radiation
#     if t == 1:
#         pass
#     else:
#         iterationTime = iterationTime2 # transient calc from the second epoch

#     # this inner cycle updates every 1 second
#     for epoch in range(iterationTime):

#         # Initialize heat flow
#         for i in range(nodelength):
#             nodes[i].radiationIn = 0
#             nodes[i].conductionIn = 0
#             nodes[i].heatInput = 0

#         # Calculate Solar radiation
#         for i in range(panelLength):
#             nodes[i].heatInput = nodes[i].heatInput + Q_in_sun[i]

#         # Detect shadow flag
#         shadowFlag = 0
#         if all(item == 90 for item in sunAnglesToPanels):
#             shadowFlag = 1

#         # Calculate Lunar albedo
#         for i in range(panelLength):
#             nodes[i].heatInput = nodes[i].heatInput + Q_in_la[i]

#         # Calculate Lunar radiation -> will be calculated as radiation heat
#         # Pe = emissivitySurface*stephanBoltz*np.power(nodes[lunarsurface].temp, 4);
#         # for i in range(panelLength):
#         #     nodes[ID].heatInput = nodes[ID].heatInput + Q_in_e[ID]

#         # Calculate Elec Power
#         electronicsPowerSum = 0;
#         if shadowFlag == 0:
#             for i in range(panelLength):
#                 nodes[i].heatInput = nodes[i].heatInput + panels[i].elecDissip
#                 electronicsPowerSum = electronicsPowerSum + panels[i].elecDissip

#         # Solar power discharging
#         if Q_in_sun[side1]*solarPanelEff*solarPanelRatio2surfaceAera*solarPowerConversionEff- electronicsPowerSum > Q_in_sun[side2]*solarPanelEff*solarPanelRatio2surfaceAera*solarPowerConversionEff:
#             nodes[side1].heatInput = nodes[side1].heatInput - electronicsPowerSum
#         elif Q_in_sun[side2]*solarPanelEff*solarPanelRatio2surfaceAera*solarPowerConversionEff- electronicsPowerSum > Q_in_sun[side1]*solarPanelEff*solarPanelRatio2surfaceAera*solarPowerConversionEff:
#             nodes[side2].heatInput = nodes[side2].heatInput - electronicsPowerSum
#         else:
#             nodes[side1].heatInput = nodes[side1].heatInput - electronicsPowerSum/2
#             nodes[side2].heatInput = nodes[side2].heatInput - electronicsPowerSum/2

#         # Radiation and conduction between nodes
#         for ID1 in range(nodelength):
#             for ID2 in range(nodelength):
#                 radsurface = connections[map(ID1,ID2,nodelength)].radSurface
#                 viewfactor = connections[map(ID1,ID2,nodelength)].viewFactor
#                 contactarea = connections[map(ID1,ID2,nodelength)].thermalContactArea
#                 contactresistatce = connections[map(ID1,ID2,nodelength)].thermalContactResistance

#                 if ID1 < panelLength and ID2 < panelLength:
#                     connections[map(ID1,ID2,nodelength)] = Connection(nodes[ID1].emissivity/10, nodes[ID2].emissivity/10, radsurface, viewfactor, contactresistatce, contactarea, 0, 0, 1, nodes[ID1].temp, nodes[ID2].temp)
#                 else:
#                     connections[map(ID1,ID2,nodelength)] = Connection(nodes[ID1].emissivity, nodes[ID2].emissivity, radsurface, viewfactor, contactresistatce, contactarea, 0, 0, 1, nodes[ID1].temp, nodes[ID2].temp)

#         # Calculate heat transfer for each node
#         for i in range(nodelength):
#             for j in range(nodelength):
#                 MonitorQrad[i][j]= connections[map(i,j,nodelength)].Qrad1
#                 MonitorQcond[i][j]= connections[map(i,j,nodelength)].Qcond1
#                 nodes[i].radiationIn = nodes[i].radiationIn + connections[map(i,j,nodelength)].Qrad1
#                 nodes[i].conductionIn = nodes[i].conductionIn + connections[map(i,j,nodelength)].Qcond1
#                 nodes[j].radiationIn = nodes[j].radiationIn + connections[map(i,j,nodelength)].Qrad2
#                 nodes[j].conductionIn = nodes[j].conductionIn + connections[map(i,j,nodelength)].Qcond2

#         # update temperature for each node
#         for i in range(nodelength):
#             nodeTemp[i][epoch] = nodes[i].temp
#             nodeHeatInput[i][epoch] = nodes[i].heatInput
#             nodeRadInput[i][epoch] = nodes[i].radiationIn
#             nodeCondInput[i][epoch] = nodes[i].conductionIn
#             nodeQtotal[i][epoch] = nodes[i].radiationIn + nodes[i].conductionIn + nodes[i].heatInput
#             if nodes[i].typeConstant == 0:
#                 nodes[i].temp = nodes[i].temp + nodeQtotal[i][epoch]/nodes[i].heatCapacitance
#             else:
#                 nodes[i].temp = nodes[i].temp

#         # if pose == 1:
#         #     print("top", nodeTemp[top][epoch], nodeHeatInput[top][epoch], nodeRadInput[top][epoch], nodeCondInput[top][epoch], nodeQtotal[top][epoch])
#         #     print("side1", nodeTemp[side1][epoch], nodeHeatInput[side1][epoch], nodeRadInput[side1][epoch], nodeCondInput[side1][epoch], nodeQtotal[side1][epoch])
#         #     print("side2", nodeTemp[side2][epoch], nodeHeatInput[side2][epoch], nodeRadInput[side2][epoch], nodeCondInput[side2][epoch], nodeQtotal[side2][epoch])
#         #     print("front", nodeTemp[front][epoch], nodeHeatInput[front][epoch], nodeRadInput[front][epoch], nodeCondInput[front][epoch], nodeQtotal[front][epoch])
#         #     print("bottom", epoch, nodeTemp[bottom][epoch], nodeHeatInput[bottom][epoch], nodeRadInput[bottom][epoch], nodeCondInput[bottom][epoch], nodeQtotal[bottom][epoch], MonitorQrad[bottom][lunarsurface])

#         # end of epoch

#     # Save data for each pose
#     for i in range(panelLength):
#         roverAllTemp[i][t][pose] = nodes[i].temp

#     # Power Generation
#     powerGenerationOfEpoch = 0
#     for i in range(side1, side2+1):
#         # print("ID=", ID)
#         powerGenerationOfEpoch = powerGenerationOfEpoch + Q_in_sun[i]*solarPanelEff*solarPanelRatio2surfaceAera*solarPowerConversionEff
#     powerConsumptionOfEpoch = electronicsPowerSum/powerConversionLoss
#     powerGen[t][pose] = powerGenerationOfEpoch-powerConsumptionOfEpoch

#     # end of pose

# print(lunarSurfaceTemp[agent_pos[0]][agent_pos[1]][t] + 273.15)
# print("top", roverAllTemp[top,t,:])
# print("side1", roverAllTemp[side1,t,:])
# print("side2", roverAllTemp[side2,t,:])
# print("front", roverAllTemp[front,t,:])
# print("rear", roverAllTemp[rear,t,:])
# print("bottom", roverAllTemp[bottom,t,:])
# print(powerGen[t,:])

# # filename = "debug.csv"
# # with open(filename, 'w', newline='') as csvfile:
# #     csvwriter = csv.writer(csvfile)
# #     csvwriter.writerows(MonitorQrad)
# #     csvwriter.writerows(MonitorQcond)
