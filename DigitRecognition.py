import pygame
import numpy as np
import matplotlib.pyplot as plt
from neuralNets import *

pygame.init()

dis = (700,500)
root = pygame.display.set_mode(dis)

f = pygame.font.SysFont("Ariel",24)

net = network([],sigmoid,sigmoid_prime)

net.loadModel("NeuralNet.net")

def surfaceToRaw(surf):
    d = 255-np.array(pygame.surfarray.array_blue(pygame.transform.smoothscale(surf,(28,28)))).T
    return np.round(d/np.max(d),1)

def getPrediction(data):
    inp = np.reshape(data,(28*28,1))
    result = net.forwardPropagate(inp)

    prediction = sorted(list(range(10)),key = lambda n: -result[n][0])

    return prediction,result.T[0]
    
def main():

    writable = pygame.Surface((400,400))
    writable.fill((255,255,255))
    writableRect = writable.get_rect()
    writableRect.topleft = (50,50)

    previousPos = [0,0]
    previousDown = False

    resetButtonText = f.render("Clear",True, (255,255,255))
    resetButtonTextRect = resetButtonText.get_rect()
    resetButtonTextRect.center = (75,50)

    resetButton = pygame.Surface((150,100))
    resetButton.fill((0,0,0))
    resetButton.blit(resetButtonText,resetButtonTextRect)
    pygame.draw.rect(resetButton,(255,255,255),((0,0),(150,100)),10)

    resetButtonRect = resetButton.get_rect()
    resetButtonRect.topleft = (500,50)

    data = np.zeros_like((28,28))
    scaledSurf = pygame.Surface((150,150))

    scaledSurfRect = scaledSurf.get_rect()
    scaledSurfRect.topleft = (500,200)

    predictionSurf = pygame.Surface((150,100))
    predictionSurfRect = predictionSurf.get_rect()
    predictionSurfRect.topleft = (500,400)

    prediction = 0

    predictionSurf.fill((0,0,0))
    predictionSurf.blit(f.render(f"{prediction}", True, (255,255,255)),(0,0))
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                
                if event.key == pygame.K_v:
                    plt.imshow(surfaceToRaw(writable))
                    plt.pause(1000)
                
        mDown = pygame.mouse.get_pressed()[0]
        mPos = pygame.mouse.get_pos()

        if mDown:
            if previousDown and writableRect.collidepoint(mPos[0],mPos[1]):
                pygame.draw.line(writable,(0,0,0),(previousPos[0]-50,previousPos[1]-50),(mPos[0]-50,mPos[1]-50),40)
                pygame.draw.circle(writable,(0,0,0),(previousPos[0]-50,previousPos[1]-51),18)
                pygame.draw.circle(writable,(0,0,0),(mPos[0]-50,mPos[1]-51),18)

            elif resetButtonRect.collidepoint(mPos[0],mPos[1]):
                writable.fill((255,255,255))

            data = surfaceToRaw(writable)
            scaledSurf = pygame.Surface((28,28))
            scaledSurf.fill((0,0,0))
            pygame.surfarray.blit_array(scaledSurf,data.T*255)
            scaledSurf = pygame.transform.scale(scaledSurf,(150,150))

            prediction = getPrediction(data)

            predictionSurf.fill((0,0,0))
            predictionSurf.blit(f.render(f"{prediction[0]}", True, (255,255,255)),(0,0))
            
            
                
        root.fill((0,0,0))

        root.blit(writable,(50,50))
        root.blit(resetButton,resetButtonRect)
        root.blit(scaledSurf,scaledSurfRect)
        root.blit(predictionSurf,predictionSurfRect)

        pygame.display.update()
        previousPos = mPos
        previousDown = mDown

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        pygame.quit()
        raise e