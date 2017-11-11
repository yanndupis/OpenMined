using System.Collections.Concurrent;
using System.Threading;
using NetMQ;
using UnityEngine;
using NetMQ.Sockets;
using static FloatTensor;

public class Job {
	public ResponseSocket socket;
	public string message;

	public Job(ResponseSocket socket_, string message_) {
		socket = socket_;
		message = message_;
	}
}

public class NetMqListener
{
    private readonly Thread _listenerWorker;

    private bool _listenerCancelled;

    public delegate void MessageDelegate(Job message);

    private readonly MessageDelegate _messageDelegate;

    private readonly ConcurrentQueue<Job> _messageQueue = new ConcurrentQueue<Job>();

	private void ListenerWork()
	{
		AsyncIO.ForceDotNet.Force();
//		using (var responseSocket = new ResponseSocket("@tcp://*:5555"))
//		{
		var responseSocket = new ResponseSocket("@tcp://*:5555");
			responseSocket.Options.ReceiveHighWatermark = 1000;

			while (!_listenerCancelled)
			{
//				var responseSocket = new ResponseSocket ("@tcp://*:5555");
				string frameString;
				if (!responseSocket.TryReceiveFrameString(out frameString)) continue;

				Debug.LogFormat("<color=purple>NetMqListener.ListenerWork: Message received {0}</color>", frameString);
				Job frameJob = new Job (responseSocket, frameString);
				_messageQueue.Enqueue(frameJob);

			}

//		}
		NetMQConfig.Cleanup();
	}

//    private void ListenerWork()
//    {
//        AsyncIO.ForceDotNet.Force();
//        using (var subSocket = new SubscriberSocket())
//        {
//            subSocket.Options.ReceiveHighWatermark = 1000;
//            subSocket.Connect("tcp://localhost:12345");
//            subSocket.Subscribe("");
//            while (!_listenerCancelled)
//            {
//                string frameString;
//                if (!subSocket.TryReceiveFrameString(out frameString)) continue;
//                Debug.Log(frameString);
//                _messageQueue.Enqueue(frameString);
//            }
//            subSocket.Close();
//        }
//        NetMQConfig.Cleanup();
//    }

    public void Update()
    {
        while (!_messageQueue.IsEmpty)
        {
            Job message;
            if (_messageQueue.TryDequeue(out message))
            {
                _messageDelegate(message);
            }
            else
            {
                break;
            }
        }
    }

    public NetMqListener(MessageDelegate messageDelegate)
    {
        _messageDelegate = messageDelegate;
        _listenerWorker = new Thread(ListenerWork);
    }

    public void Start()
    {
        _listenerCancelled = false;
        _listenerWorker.Start();
    }

    public void Stop()
    {
        _listenerCancelled = true;
        _listenerWorker.Join();
    }
}

public class ClientObject : MonoBehaviour
{
    private NetMqListener _netMqListener;

	private SyftController controller;

	[SerializeField]
	private ComputeShader shader;

    private void HandleMessage(Job job_with_message)
    {
		
//        var splittedStrings = message.Split(' ');
//        if (splittedStrings.Length != 3) return;
//        var x2 = float.Parse(splittedStrings[0]);
//        var y2 = float.Parse(splittedStrings[1]);
//        var z2 = float.Parse(splittedStrings[2]);
//        transform.position = new Vector3(x2, y2, z2);

//		controller.processMessage (job_with_message.message);
//		Debug.Log ("Message Processed!!");

		job_with_message.socket.SendFrame ("Done!");
		Debug.Log ("Response Sent!!");

    }

    private void Start()
    {

        _netMqListener = new NetMqListener(HandleMessage);
        _netMqListener.Start();

		controller = new SyftController (shader);
    }

    private void Update()
    {
        _netMqListener.Update();
    }

    private void OnDestroy()
    {
        _netMqListener.Stop();
    }
}
