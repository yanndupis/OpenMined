using UnityEngine;
using System.Collections;
using Newtonsoft.Json.Linq;

namespace OpenMined.Network.Servers.Ipfs
{
    public class IpfsJobConfig
    {
        [SerializeField] public float lr;

        public IpfsJobConfig(float lr)
        {
            this.lr = lr;
        }
    }
}
