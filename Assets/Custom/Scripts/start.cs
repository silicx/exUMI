using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;
using Unity.XR.Oculus;
using Meta.XR.Depth;
using TMPro;
using UnityEditor;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Net.NetworkInformation;
using System.Text;

public class StartScene : MonoBehaviour
{
    // Start is called before the first frame update
    OVRCameraRig cameraRig;
    private TextMeshProUGUI init_text;
    public static string pc_ip;
    public static string local_ip;

    private TouchScreenKeyboard overlayKeyboard;
    
    void Start()
    {
        init_text = GameObject.Find("StartText").GetComponent<TextMeshProUGUI>();
        GetLocalIPAddress();
        overlayKeyboard = TouchScreenKeyboard.Open("192.168.1.128", TouchScreenKeyboardType.Default);
    }

    public void GetLocalIPAddress()
    {
        var host = Dns.GetHostEntry(Dns.GetHostName());
        foreach (var ip in host.AddressList)
        {
            if (ip.AddressFamily == AddressFamily.InterNetwork)
            {
                local_ip = ip.ToString();
            }
        }
    }


    // Update is called once per frame
    void Update()
    {
        if (overlayKeyboard != null && overlayKeyboard.status != TouchScreenKeyboard.Status.Visible)
        {
            pc_ip = overlayKeyboard.text;
        }

        init_text.text = "VR: " + local_ip + "\nPC: " + pc_ip + "\nAny key to continue";

        if (OVRInput.GetUp(OVRInput.RawButton.Y) || OVRInput.GetUp(OVRInput.RawButton.X) || OVRInput.GetUp(OVRInput.RawButton.A) || OVRInput.GetUp(OVRInput.RawButton.B))
        {
            SceneManager.LoadScene("HandSelect");
        }
    }
}
