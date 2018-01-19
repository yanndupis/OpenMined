using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine.Networking;
using Newtonsoft.Json;

namespace OpenMined.Network.Servers.BlockChain.Requests
{
    public abstract class BaseRequest<T>
    {

        // public static string URL = "http://localhost:3000/";

        public Method method;
        public string path;

        protected string response;
            
        public BaseRequest (Method method, string path)
        {
            this.method = method;
            this.path = path;
        }

        protected UnityWebRequest GetRequest (string subPath = "", WWWForm postBody = null)
        {

            var builder = new UriBuilder();
            builder.Host = "192.168.2.14";
            builder.Port = 3000;
            builder.Scheme = "http";
            builder.Path = this.path + subPath;

            switch (this.method)
            {
                case Method.GET:
                    return UnityWebRequest.Get(builder.Uri.AbsoluteUri);
                case Method.POST:
                    var req = UnityWebRequest.Post(builder.Uri.AbsoluteUri, postBody);
                    //req.SetRequestHeader("content-type", "application/json");
                    return req;
            };

            return null;
        }

        public IEnumerator RunRequest ()
        {
            var request = GetWebRequest();
            yield return request.SendWebRequest();
            if (request.isNetworkError || request.isHttpError)
            {
                Debug.Log(request.error);
                yield return null;
            }
            else
            {
                yield return request.downloadHandler.text;
                this.response = request.downloadHandler.text;
            }
        }

        public void RunRequestSync ()
        {
            var request = GetWebRequest();
            request.SendWebRequest();

            while (!request.isDone) {}

            if (request.isNetworkError || request.isHttpError)
            {
                Debug.Log(request.error);
            }
            else
            {
                this.response = request.downloadHandler.text;
            }
        }

        public T GetResponse ()
        {
            return JsonConvert.DeserializeObject<T>(this.response);
        }

        abstract public UnityWebRequest GetWebRequest();
    }

    public enum Method
    {
        GET,
        POST
    }
}
