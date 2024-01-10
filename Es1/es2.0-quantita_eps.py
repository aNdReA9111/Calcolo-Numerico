'''
    F(2,53,−1024,1023) precisione doppia: 64 bit. Le cifre della
    mantissa sono 53 (rappresentati 52 bit) e dell’esponente 11
    (211 = 2048 =U-L+1; con L=-1024 e U=1023)

    eps = 1/2 * B^(1-t)

    eps è il più piccolo numero macchina positivo
    tale che fl(1+eps) > 1

    esponente rappresenta "t" le cifre della mantissa per il
    sistema floating point a precisione doppia

    praticamente il codice sfrutta il fatto di sforare
    le cifre della mantissa prefissate per il tipo di dato
    con virgola mobile a precisione doppia e quindi ottenere
    il più piccolo numero macchina e la dimensione della
    mantissa stessa
'''
import numpy as np
mantissa = 1   #t
eps = np.float32(1)

B=np.float32(2) #base

while np.float32(1)+eps/B>np.float32(1):
    eps = eps/B
    mantissa += 1

print(
      "eps:", eps,
      "\nmantissa (t):", mantissa-1,
    )
mantiissa = 1   #t
eps = 1
B=2 #base
while 1+eps>1:
    eps/=B
    mantissa += 1
print("eps:", eps, "\nmantissa (t):", mantissa-1)

print("Mantissa per float16:", np.finfo(np.float16).nmant)
print("Mantissa per float32:", np.finfo(np.float32).nmant)
print("Mantissa per float64:", np.finfo(np.float64).nmant)

print("Epsilon di macchina per float16:", np.finfo(np.float16).eps)
print("Epsilon di macchina per float32:", np.finfo(np.float32).eps)
print("Epsilon di macchina per float64:", np.finfo(np.float64).eps)

