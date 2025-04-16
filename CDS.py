import numpy as np
import matplotlib.pyplot as plt

phi_s = 0.5
rho = 2
l = 3
gamma = 3
n = 50
dx = l/n
dy = l/n

def u(x,y):
    return [x**2+1, y**2+1]

phi_grid = np.zeros((n,n))
A = np.zeros((n**2,n**2))
B = np.zeros((n**2,1))

for i in range(n):
    for j in range(n):
        row = n * i + j

        x,y = j*l/n+dx/2, i*l/n+dy/2
        De, Dw, Dn, Ds = gamma * dy/dx, gamma * dy/dx, gamma * dx/dy, gamma * dx/dy
        Fe, Fw, Fn, Fs = rho*u(x + dx/2,y)[0]*dy, rho*u(x - dx/2,y)[0]*dy, rho*u(x,y + dy/2)[1]*dx, rho*u(x,y - dy/2)[1]*dx
        Sc = phi_s
        # delV = dx**2
        aE = De - Fe / 2
        aW = Dw + Fw / 2
        aN = Dn - Fn / 2
        aS = Ds + Fs / 2
        b = Sc*dx*dy

        #boundaryconditions
        if i==0:
            #south
            phibc = 1
            b += (2*Ds + Fs) * phibc
            aS = 0
            Fs = 0
            Ds = 2 * Ds

        if j==0:
            # west
            phibc = 0
            b += (2 * Dw + Fw) * phibc
            aW = 0
            Fw = 0
            Dw = 2 * Dw

        if j==n-1:
            #east
            aE = 0
            Fe = 2*Fe
            De = 0

        if i==n-1:
            #north
            aN = 0
            Fn = 2*Fn
            Dn = 0


        aP = (De + Fe / 2) + (Dw - Fw / 2) + (Dn + Fn / 2) + (Ds - Fs / 2)
        A[row,row]+=aP

        if j!=0:
            A[row, row-1] += -aW
        if j!=n-1:
            A[row, row+1] += -aE
        if i!=0:
            A[row,row-n] += -aS
        if i!=n-1:
            A[row, row+n] += -aN

        B[row,0] += b

# phi = np.dot(np.linalg.inv(A), B)
phi = np.linalg.solve(A,B)
# phi = gauss_seidel(A,B)
phi = phi.reshape(n, n)
print(phi)
plt.imshow(phi, interpolation='none', cmap='viridis', origin='lower')
plt.colorbar(label='phi')
plt.title('Discrete Color Grid')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
midsection = phi[n//2,:]
xaxis = np.linspace(0,l,n)
plt.plot(xaxis, midsection)
plt.show()