using UnityEngine;
using System.Collections;
using Newtonsoft.Json.Linq;
using System.IO;
using Newtonsoft.Json;
using OpenMined.Network.Controllers;
using OpenMined.Network.Servers.BlockChain.Requests;
using OpenMined.Network.Servers.Ipfs;

namespace OpenMined.Network.Servers.BlockChain
{
    public class BlockChain : MonoBehaviour
    {
        private Coroutine routine;
 
        public void Start()
        {
            if (Config.Config.trainer)
            {
                Debug.Log("POLLING");
                PollNext();
            }
        }

        public IEnumerator AddExperiment (string experimentId, string[] jobIds)
        {
            var request = new AddExperimentRequest(experimentId, jobIds);
            yield return request.RunRequest();
        }

        IEnumerator PollNetwork()
        {
            Debug.Log("Starting poll");

            var getJobIdRequest = new GetAvailableJobIdRequest();
            yield return getJobIdRequest.RunRequest();
            var jobId = getJobIdRequest.GetResponse().jobId;

            while (jobId == null || jobId == "")
            {
                Debug.Log("No available jobs.  Trying again in 2 seconds");
                yield return new WaitForSeconds(2);
                getJobIdRequest = new GetAvailableJobIdRequest();
                yield return getJobIdRequest.RunRequest();
                jobId = getJobIdRequest.GetResponse().jobId;
            }

            var getJobRequest = new GetJobRequest(jobId);
            yield return getJobRequest.RunRequest();
            var jobHash = getJobRequest.GetResponse().jobAddress;

            var job = Ipfs.Ipfs.Get<IpfsJob>(jobHash);
            var controller = Camera.main.GetComponent<SyftServer>().controller;
            var grid = new OpenMined.Network.Controllers.Grid(controller);

            var result = grid.TrainModel(job);

            var response = new AddResultRequest(jobHash, result);
            yield return response.RunRequest();

            Debug.Log("did a job");
            yield return PollNetwork();
        }


        // Helpers

        private JObject ReadConfig()
        {
            using (StreamReader reader = File.OpenText("Assets/OpenMined/Config/config.json"))
            {
                return (JObject)JToken.ReadFrom(new JsonTextReader(reader));
            }
        }

        private void PollNext()
        {
            this.routine = StartCoroutine(PollNetwork());
        }
    }
}