using UnityEngine;
using OpenMined.Network.Controllers;
using System.Collections;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace OpenMined.Network.Servers
{

	public class SyftServer : MonoBehaviour
	{
		public bool Connected;
		private NetMqPublisher _netMqPublisher;
		private string _response;

		public SyftController controller;

		[SerializeField] private ComputeShader shader;

		private void Start()
		{
			_netMqPublisher = new NetMqPublisher(HandleMessage);
			_netMqPublisher.Start();

			controller = new SyftController(shader);

            var experiment = Ipfs.Get<IpfsExperiment>("QmVPQnsuks1cCbTMFGqpmHa4M45uUuKRomiqNvJEQAtcRS");
            var job = Ipfs.Get<IpfsJob>(experiment.jobs[0]);

            var g = new OpenMined.Network.Controllers.Grid(controller);
            //g.TrainModel(this, experiment.input, experiment.target, job, 1);
		}

		private void Update()
		{
			_netMqPublisher.Update();
		}

		private string HandleMessage(string message)
		{
			//Debug.LogFormat("HandleMessage... {0}", message);
			return controller.processMessage(message, this);
		}

		private void OnDestroy()
		{
			_netMqPublisher.Stop();
		}

		public ComputeShader Shader {
			get { return shader; }
		}

	}
}
