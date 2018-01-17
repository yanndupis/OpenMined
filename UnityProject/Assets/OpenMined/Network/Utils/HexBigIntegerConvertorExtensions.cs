using System;
using System.Linq;
using Org.BouncyCastle.Math;

namespace OpenMined.Hex.HexConvertors.Extensions
{
    public static class HexBigIntegerConvertorExtensions
    {
        public static byte[] ToByteArray(this BigInteger value, bool littleEndian)
        {
            byte[] bytes;
            if (BitConverter.IsLittleEndian != littleEndian)
            {
                bytes = value.ToByteArray().Reverse().ToArray();
            }
            else
            {
                bytes = value.ToByteArray().ToArray();
            }
            return bytes;
        }

        public static string ToHex(this BigInteger value, bool littleEndian, bool compact=true)
        {
            if(value.SignValue < 0) throw new Exception("Hex Encoding of Negative BigInteger value is not supported");
            if (value == BigInteger.Zero) return "0x0";

            byte[] bytes = value.ToByteArray(littleEndian);

            if(compact)
            return "0x" + bytes.ToHexCompact();

            return "0x" + bytes.ToHex();
        }


        public static BigInteger HexToBigInteger(this string hex, bool isHexLittleEndian)
        {
            if (hex == "0x0") return BigInteger.Zero;

            var encoded = hex.HexToByteArray();

            if ((BitConverter.IsLittleEndian != isHexLittleEndian))
            {
                var listEncoded = encoded.ToList();
                listEncoded.Insert(0, 0x00);
                encoded = listEncoded.ToArray().Reverse().ToArray();

            }
            return new BigInteger(encoded);
        }

        
    }
}