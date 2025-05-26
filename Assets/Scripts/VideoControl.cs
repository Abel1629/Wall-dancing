using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.Diagnostics;
using System.IO;

public class VideoControl : MonoBehaviour
{
    private TcpClient client;
    private NetworkStream stream;
    private byte[] buffer = new byte[1024];

    private string host = "127.0.0.1";
    private int port = 65432;

    private bool isConnected = false;
    private float reconnectDelay = 2.0f;
    private float reconnectTimer = 0f;

    private static VideoControl instance;

    private PoseSwitcher poseSwicher;

    private Process pythonProcess;

    void Start()
    {
        StartPythonScript();
        TryConnect();
    }

    void Awake()
    {
        if (instance != null && instance != this)
        {
            Destroy(gameObject); // Prevent duplicates
            return;
        }

        instance = this;
        DontDestroyOnLoad(gameObject);

        SceneManager.sceneLoaded += OnSceneLoaded; // Register scene load event
    }

    private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        poseSwicher = FindObjectOfType<PoseSwitcher>();
        if (poseSwicher == null)
        {
            UnityEngine.Debug.LogWarning("PoseSwitcher not found in scene: " + scene.name);
        }
        else
        {
            UnityEngine.Debug.Log("PoseSwitcher found in scene: " + scene.name);
        }
    }

    void Update()
    {
        if (isConnected)
        {
            try
            {
                if (stream.DataAvailable)
                {
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    string poseStr = Encoding.UTF8.GetString(buffer, 0, bytesRead).Trim();
                    UnityEngine.Debug.Log("Current pose number " + poseStr);
                    UpdateCharacterPose(poseStr);
                }
            }
            catch (Exception ex)
            {
                UnityEngine.Debug.LogWarning("Connection lost: " + ex.Message);
                Disconnect();
            }
        }
        else
        {
            reconnectTimer += Time.deltaTime;
            if (reconnectTimer >= reconnectDelay)
            {
                reconnectTimer = 0f;
                TryConnect();
            }
        }
    }


    private void StartPythonScript()
{
    // Path to the compiled .exe file
    string exePath = Path.Combine(Application.dataPath, "../Python/pose_detection.exe");

    ProcessStartInfo startInfo = new ProcessStartInfo
    {
        FileName = exePath,
        UseShellExecute = false,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        CreateNoWindow = true
    };

    try
    {
        pythonProcess = new Process();
        pythonProcess.StartInfo = startInfo;
        pythonProcess.OutputDataReceived += (sender, args) => UnityEngine.Debug.Log(args.Data);
        pythonProcess.ErrorDataReceived += (sender, args) => UnityEngine.Debug.LogError(args.Data);

        pythonProcess.Start();
        pythonProcess.BeginOutputReadLine();
        pythonProcess.BeginErrorReadLine();

        UnityEngine.Debug.Log("pose_detection.exe started.");
    }
    catch (Exception e)
    {
        UnityEngine.Debug.LogError("Failed to start pose_detection.exe: " + e.Message);
    }
}

    void TryConnect()
    {
        try
        {
            client = new TcpClient();
            client.Connect(host, port);
            stream = client.GetStream();
            isConnected = true;
            UnityEngine.Debug.Log("Connected to Python server.");
        }
        catch (Exception e)
        {
            UnityEngine.Debug.Log("Connection attempt failed. Retrying...");
            UnityEngine.Debug.Log(e.Message);
            isConnected = false;
        }
    }

    void UpdateCharacterPose(string pose)
    {
        if (poseSwicher != null)
        {
            poseSwicher.pose = pose;
            UnityEngine.Debug.Log("Updated pose: " + pose);
        }
        else
        {
            UnityEngine.Debug.LogWarning("PoseSwitcher not assigned yet.");
        }
    }

    void OnApplicationQuit()
    {
        // Close network connection
        Disconnect();

        // Kill the Python process if it's still running
        if (pythonProcess != null && !pythonProcess.HasExited)
        {
            pythonProcess.Kill();
            pythonProcess.Dispose();
            UnityEngine.Debug.Log("Python process terminated.");
        }

        // Clean up leftover _MEIxxxxx folders from PyInstaller
        try
        {
            string tempPath = Path.GetTempPath();
            string[] meiFolders = Directory.GetDirectories(tempPath, "_MEI*");

            foreach (string folder in meiFolders)
            {
                try
                {
                    Directory.Delete(folder, true);
                    UnityEngine.Debug.Log($"Deleted temp folder: {folder}");
                }
                catch (Exception e)
                {
                    UnityEngine.Debug.LogWarning($"Could not delete {folder}: {e.Message}");
                }
            }
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogWarning("Error while cleaning temp folders: " + e.Message);
        }
    }

    void Disconnect()
    {
        isConnected = false;
        stream?.Close();
        client?.Close();
    }

}
