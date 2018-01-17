using UnityEngine;
using System.Collections;
using Newtonsoft.Json.Linq;
using System.IO;
using Newtonsoft.Json;
using OpenMined.Network.Controllers;

namespace OpenMined.Network.Servers
{
    public class BlockChain : MonoBehaviour
    {
        private Coroutine routine;
 
        public void Start()
        {
            var o = ReadConfig();

            if (o["trainer"].ToObject<bool>())
            {
                Debug.Log("POLLING");
                PollNext();
            }
        }

        void PollNext()
        {
            this.routine = StartCoroutine(PollNetwork());
        }

        IEnumerator PollNetwork()
        {
            var request = new Request();

            yield return request.GetBlockNumber(this);
            yield return request.GetModel(this);
            
            var ipfsAddress = request.modelResponse.configAddress;
            var owner = request.modelResponse.owner;

            Debug.Log("IPFS address: " + ipfsAddress);
            Debug.Log("owner: " + owner);

            IpfsModel model = Ipfs.GetModel(ipfsAddress);
            if (model != null)
            {
                Debug.Log("Got the IpfsModel: " + model.input);
                
                // TODO do some training!!!
                //var g = new Controllers.Grid(controller);
                //g.TrainModel(model, numModels - 1);
            }

            Debug.Log("Blockchain polled");

            // TODO should probably only poll again once above training is done
            yield return new WaitForSeconds(10);
            PollNext();
        }
        
        JObject ReadConfig()
        {
            using (StreamReader reader = File.OpenText("Assets/OpenMined/Config/config.json"))
            {
                return (JObject)JToken.ReadFrom(new JsonTextReader(reader));
            }
        }
    }
}