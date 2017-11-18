OpenMined Unity Application
=============================================
## Introduction
OpenMined Unity Application applies the [PySyft](https://github.com/OpenMined/PySyft) library into a Unity Application. Please see the PySyft repository README.md for more details on the intent of OpenMined and to familiarize yourself more with the basic concepts of the project.

## Quick Setup

1. Download Unity from [here](https://store.unity.com/). I chose the personal version. This will provide you a .dmg installer, which will download and install the necessary components (~800mb). 
    - If you do not already have one, you will have to create a Unity account when you open the Application for the first time.
2. Open project in Unity `[File -> Open Project -> Directory/To/OpenMined`]

3. Open Juptyer Notebooks in the `notebooks` directory.  

## Setup Troubleshooting 

If you have an issue, refer to the following steps for a more detailed project setup. These steps were confirmed to work under a Windows enviornment, though the steps should be applicable to alternative operating systems. 

The steps come in two parts:   
**Part 1:** Unity Setup   
**Part 2:** Jupyter Setup

**Unity Setup**  

1. Download Unity from [here](https://store.unity.com/). I chose the personal version. This will provide you a .dmg installer, which will download and install the necessary components (~800mb).  
2. Open project in Unity `[Open(Top Right of Home Screen)	 -> Directory/To/OpenMined`]
3. In the Project Pane (usually below), Double Click Assets/_Scenes/DefaultScene. If you can't find the file for some reason:
- On the right menu: Check `Main Camera` object has `SyftServer.cs` component attached to it
- On the bottom dialog: Go to `Assets/OpenMined/Network/Servers` drag `SyftServer.cs` to `Main Camera` object
- Add a `Compute Shader` to the `Shader` variable of `SyftServer.cs` script
- Go to `Assets/OpenMined/Syft/Math/Shaers` drag `NewComputeShader` to `SyftServer (Script)` component recently attached to `Main Camera`
4. Hit `Play` on the Unity Editor

**Jupyter Setup**

1. Open `basic-python-network-gpu.ipynb` 
2. Run the Jupyter Notebook

### For OSX (High Sierra)

Same steps as above. Download Unity from [here](https://store.unity.com/). I chose the personal version. This will provide you a .dmg installer, which will download and install the necessary components (~800mb). 

## General Troubleshooting

1) *If my applications do not seem to be communicating between eachother...*

**Check if the Server is running...**
___    
It should run on port 5555 and this can be checked by running the following command on CMD with administrator permissions.  
```
netstat -a -b | findstr :5555  
```  
If just the Server is running, the output should be:  
```
TCP    0.0.0.0:5555           YOUR_PC_NAME:0      LISTENING
```  
If both Server and Jupyter Notebook are running and communicating, the output should be:  

```
TCP    0.0.0.0:5555           YOUR_PC_NAME:0      LISTENING
TCP    127.0.0.1:5555         YOUR_PC_NAME:63956  ESTABLISHED
TCP    127.0.0.1:63956        YOUR_PC_NAME:5555   ESTABLISHED
```  
Another way:   
**Osx/Linux**  

```  
lsof -i :5555
```  
Result should be:   

```  
Unity   1709 user   38u  IPv4 0x59e297c6d0d734e31      0t0  TCP *:personal-agent (LISTEN)
```
---

2) *My application randomly stops working.*   

**Jupyter Notebook only works if Unity has focus**  
	By default, the "Run in background" options is disabled. So if the Unity Editor loses focus then the Jupyter Notebook won't work.
Go to Edit -> Project Settings -> Player. The inspector pane will now change to show the player settings. Look for the option that says "Run In Background" and check it [1]

### References

[1] [stop unity pausing when it loses focus](https://answers.unity.com/questions/42509/stop-unity-pausing-when-it-loses-focus.html)

