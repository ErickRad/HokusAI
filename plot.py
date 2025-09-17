import matplotlib.pyplot as plt
import properties
import torch
import pynvml

class LossPlotter:
    def __init__(self, delta=15):
        self.epochs = []
        self.trainLosses = []
        self.vramUsages = []
        self.temperatures = []
        self.gpuUsages = []
        self.delta = delta
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
    def getGpuStats(self):
        memInfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        vramUsedGb = memInfo.used / 1024 ** 3
        vramTotalGb = memInfo.total / 1024 ** 3
        vramPercent = vramUsedGb / vramTotalGb * 100

        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu

        return vramPercent, vramUsedGb, temp, util
    
    def add(self, epoch, trainLoss):
        self.epochs.append(epoch)
        self.trainLosses.append(trainLoss)

        vramPercent, vramGb, temp, util = self.getGpuStats()
        self.vramUsages.append(vramPercent)

        self.temperatures.append(temp)
        self.gpuUsages.append(util)
        self.vramGb = vramGb

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(self.epochs, self.trainLosses, label='Train Loss', color='blue')
        
        if self.epochs:
            ax1.plot(self.epochs[-1], self.trainLosses[-1], 'o', color='blue')
            ax1.annotate(f"{self.trainLosses[-1]:.4f}",
                         xy=(self.epochs[-1], self.trainLosses[-1]),
                         xytext=(-10, 8), textcoords='offset points',
                         ha='right', color='blue')
            
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        ax1.legend()    
        ax1.grid(True)
        
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        start = max(0, len(self.epochs) - self.delta)
        
        epochs_window = self.epochs[start:]
        
        ax2.plot(epochs_window, self.vramUsages[start:], label='VRAM %', color='purple')
        ax2.plot(epochs_window, self.gpuUsages[start:], label='GPU %', color='green')
        ax2.plot(epochs_window, self.temperatures[start:], label='Temp °C', color='red')
        
        if epochs_window:
            ax2.plot(self.epochs[-1], self.vramUsages[-1], 'o', color='purple')
            ax2.plot(self.epochs[-1], self.gpuUsages[-1], 'o', color='green')
            ax2.plot(self.epochs[-1], self.temperatures[-1], 'o', color='red')
            ax2.annotate(f"{self.vramGb:.2f} GB ({self.vramUsages[-1]:.1f}%)",
                         xy=(self.epochs[-1], self.vramUsages[-1]),
                         xytext=(0, 8), textcoords='offset points', ha='center', color='purple')
            ax2.annotate(f"{self.gpuUsages[-1]:.1f}%", xy=(self.epochs[-1], self.gpuUsages[-1]),
                         xytext=(0, 8), textcoords='offset points', ha='center', color='green')
            ax2.annotate(f"{self.temperatures[-1]:.1f}°C", xy=(self.epochs[-1], self.temperatures[-1]),
                         xytext=(0, 8), textcoords='offset points', ha='center', color='red')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('%')
        
        ax2.set_ylim(75, max(max(self.temperatures[-self.delta:]) + 10, 100))
        ax2.legend()
        
        ax2.grid(True)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        plt.savefig(f'{properties.LOGS_PATH}/metrics.png', bbox_inches='tight')
        plt.close()