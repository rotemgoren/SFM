# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 10:42:31 2018

@author: Rotem Goren
"""
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

#img1_u = cv2.undistort(img1, mtx, dist)
#img2_u = cv2.undistort(img2, mtx, dist)

class Graph:
    def __init__(self):
        super(Graph, self).__init__()
        self.Nodes=list([])
        self.Edges=list([])


    def updateNodes(self,kp,des,P):
        self.Nodes.append({'kp':kp, 'des':des,'P':P})

    def updateEdges(self,newEdges):
        if(self.Edges==[]):
            self.Edges=newEdges
        else:
            M=len(self.Edges)
            for newEdge in newEdges:
                connected=False
                for i in range(M):
                    if(self.Edges[i]['featureInx'][-1]==newEdge['featureInx'][0] and self.Edges[i]['viewID'][-1]==newEdge['viewID'][0]):
                        self.Edges[i]['featureInx'].append(newEdge['featureInx'][1])
                        self.Edges[i]['viewID'].append(newEdge['viewID'][1])
                        connected=True
                        break
                if(connected==False):
                    self.Edges.append(newEdge)
    def sortAndRemove(self):
        newEdges=[]
        for edge in self.Edges:
            if(len(edge['viewID'])>2):
                newEdges.append(edge)

        self.Edges=newEdges
        print(self.Edges)


def Rot2quad(R):
    q=np.zeros(4)

    q[0] =  R[0, 0] - R[1, 1] - R[2, 2] + 1
    q[1] = -R[0, 0] + R[1, 1] - R[2, 2] + 1
    q[2] = -R[0, 0] - R[1, 1] + R[2, 2] + 1
    q[3] =  R[0, 0] + R[1, 1] + R[2, 2] + 1
    val=np.max(q)
    inx=np.argmax(q)
    if(inx==0):
        q[0] = 0.5*np.sqrt(val)
        q[1] = (R[0,1]-R[1,0])/(4*q[0])
        q[2] = (R[0,2]-R[2,0])/(4*q[0])
        q[3] = (R[1,2]-R[2,1])/(4*q[0])

    elif(inx==1):
        q[1] = 0.5 * np.sqrt(val)
        q[0] = (R[0, 1] - R[1, 0]) / (4 * q[1])
        q[2] = (R[1, 2] - R[2, 1]) / (4 * q[1])
        q[3] = (R[2, 0] - R[0, 2]) / (4 * q[1])

    elif(inx==2):
        q[2] = 0.5 * np.sqrt(val)
        q[0] = (R[2, 0] - R[0, 2]) / (4 * q[2])
        q[1] = (R[1, 2] - R[2, 1]) / (4 * q[2])
        q[3] = (R[0, 1] - R[1, 0]) / (4 * q[2])

    elif(inx==3):
        q[3] = 0.5 * np.sqrt(val)
        q[0] = (R[1, 2] - R[2, 1]) / (4 * q[3])
        q[1] = (R[2, 0] - R[0, 2]) / (4 * q[3])
        q[2] = (R[0, 1] - R[1, 0]) / (4 * q[3])

    return q

def quad2Rot(q):
    x,y,z,w=q
    R=np.array([[1-2*y**2-2*z**2,2*x*y+2*z*w,2*x*z-2*y*w],
                [2*x*y-2*z*w, 1-2*x**2-2*z**2,2*y*x+2*x*w],
                [2*x*z+2*y*w,2*y*z-2*x*w,1-2*x**2-2*y**2]])
    return R
def bundle_adjust(Graph,X,K):

    mue=1e-9
    for _ in range(2):
        delta_Rc=np.zeros((len(Graph.Nodes),12))
        count=np.zeros((len(Graph.Nodes)))
        M=len(Graph.Nodes) # Num of views
        N=len(Graph.Edges) # Num of 3D points
        J=np.zeros((2*(M*N),7*M+3*N))
        e = np.array([])
        K=np.eye(3)

        for i,edge in enumerate(Graph.Edges):
            X_ = X[:, i]
            viewID = edge['viewID']
            featureInx = edge['featureInx']

            for j in range(len(viewID)):
                b = np.zeros(2)
                b[0], b[1] = Graph.Nodes[viewID[j]]['kp'][featureInx[j]].pt

                R = Graph.Nodes[viewID[j]]['P'][0:3, 0:3]
                q=Rot2quad(R)
                #R=quad2Rot(q)
                t = Graph.Nodes[viewID[j]]['P'][0:3, -1]

                #theta=np.concatenate((theta,c))
                #theta = np.concatenate((theta, np.reshape(R,9)))



                m = K @ (R @ X_ + t)
                f = np.array([m[0] / m[2], m[1] / m[2]])

                dm_dR = np.array( [X_[0]*K[:,0],X_[1]*K[:,0],X_[2]*K[:,0],X_[0]*K[:,1],X_[1]*K[:,1],X_[2]*K[:,1],X_[0]*K[:,2],X_[1]*K[:,2],X_[2]*K[:,2]]).T
                dR_dq=np.array([[0,-4*q[1],-4*q[2],0],
                                [2*q[1],2*q[0],2*q[3],2*q[2]],
                                [2*q[2],-2*q[3],2*q[0],-2*q[1]],
                                [2*q[1],2*q[0],-2*q[3],-2*q[2]],
                                [-4*q[0],0,-4*q[2],0],
                                [2*q[3],2*q[2],2*q[1],2*q[0]],
                                [2*q[2],2*q[3],2*q[0],2*q[1]],
                                [-2*q[3],2*q[2],2*q[1],-2*q[0]],
                                [-4*q[0],-4*q[1],0,0]])

                dm_dt = K
                dm_dX = K @ R

                df_dt = np.array(
                    [(m[2] * dm_dt[0] - m[0] * dm_dt[2]) / m[2] ** 2, (m[2] * dm_dt[1] - m[1] * dm_dt[2]) / m[2] ** 2])
                df_dX = np.array(
                    [(m[2] * dm_dX[0] - m[0] * dm_dX[2]) / m[2] ** 2, (m[2] * dm_dX[1] - m[1] * dm_dX[2]) / m[2] ** 2])
                df_dR = np.array(
                    [(m[2] * dm_dR[0] - m[0] * dm_dR[2]) / m[2] ** 2, (m[2] * dm_dR[1] - m[1] * dm_dR[2]) / m[2] ** 2])
                df_dq=df_dR @ dR_dq


                J[2*M*i+2*j:2*M*i+2*(j+1),7*viewID[j]:7*(viewID[j]+1)]=np.concatenate((df_dq.T, df_dt.T)).T
                J[2*M*i+2*j:2*M*i+2*(j+1), 7*M+3*i:7*M+3*(i+1)]=df_dX


                e=np.hstack((e,(b-f).T))

            # delta = np.linalg.inv(J.T @ J + lam * np.diag(J.T @ J) * np.eye(J.shape[1])) @ J.T @ e
            # delta_X=delta[:3]
            # X[:, i]+=delta_X
            #
            # for j, id in enumerate(viewID):
            #     delta_Rc[id]+=delta[3+12*j:3+12*(j+1)]
            #     count[id]+=1

        inx=np.where(np.sum(J, 1) == 0)[0]
        J=np.delete(J,inx,axis=0)

        A=J.T @ J
        V=A[-3*N:,-3*N:]
        U=A[:7*M,:7*M]
        W=A[:7*M,-3*N:]

        inv_V=np.linalg.inv(V)

        B1=np.hstack((np.eye(len(U)), -W @ inv_V))
        B2=np.hstack((np.zeros((len(V), len(U))), np.eye(len(V))))
        B=np.vstack((B1,B2))

        eps=B @ (J.T @ e)

        eps_a = eps[:(7*M)]
        eps_b = eps[7*M:]


        delta_a = (U-W @ inv_V @ W.T) @ eps_a
        delta_b = inv_V @ (eps_b-W.T @ delta_a)


        for i in range(M):
            delta_M=mue*delta_a[7*i:7*(i+1)]
            dq=delta_M[0:4]
            dt=delta_M[4:]
            R = Graph.Nodes[i]['P'][0:3, 0:3]
            q = Rot2quad(R)
            q=q+dq
            q=q/np.sum(q**2)
            R=quad2Rot(q)
            t=Graph.Nodes[i]['P'][0:3, -1]
            t=np.array([t+dt])
            P=np.concatenate((R,t.T),axis=1)
            Graph.Nodes[i]['P'] = P


        for i in range(N):
            dX=mue*delta_b[3*i:3*(i+1)]
            X[:,i]=X[:,i]+dX

        norm_e=np.norm(e)
        if(prev_e<=norm_e):
            break;
        



def find_correspondence_points(kp1,des1,kp2,des2,viewID1,viewID2):

    FLANN_INDEX_KTREE=0
    index_params=dict(algorithm=FLANN_INDEX_KTREE,trees=5)
    search_params=dict(checks=50)
    
    flann=cv2.FlannBasedMatcher(index_params,search_params)
    matches=flann.knnMatch(des1,des2,k=2)
    
    #matchesMask = [[0,0] for i in range(len(matches))]
    pts1=[]
    pts2=[]
    
    good=[]
    for i,(m,n) in enumerate(matches):
        if m.distance <0.75* n.distance:
            good.append(m)

    matchInx=[{'featureInx' :[m.queryIdx ,m.trainIdx] ,'viewID':[viewID1,viewID2]} for m in good]
    pts1=np.asarray([kp1[m.queryIdx].pt for m in good])
    pts2=np.asarray([kp2[m.trainIdx].pt for m in good])
            #pts2.append(kp2[m.trainIdx].pt)
            #pts1.append(kp1[m.queryIdx].pt)
            #matchesMask[i]=[1,0]
           
    #pts1=np.int32(pts1)
    #pts2=np.int32(pts2)

    '''
    retval,mask = cv2.findHomography(pts1,pts2,cv2.RANSAC,100.0)
    mask=mask.ravel()

    new_macthInx=[]
    for i in range(len(mask)):
        if(mask[i]==1):
            new_macthInx.append(matchInx[i])


    pts1=pts1[mask==1]
    pts2=pts2[mask==1]
    '''
    #draw_params = dict(matchColor = (0,255,0),
    #                   singlePointColor = (255,0,0),
    #                   matchesMask = matchesMask,
    #                   flags = 0)
    
    return pts1,pts2,matchInx

#img3 = cv2.drawMatchesKnn(img1_u,kp1,img2_u,kp2,matches,None, flags=2)

#plt.imshow(img3,),plt.show()

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    r,c,_ = img1.shape
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2



def compute_P_from_essential(E):
    """ Compute the second camera matrix (assuming P1 = [I 0])
        from an essential matrix. E = [t]R
    :returns: list of 4 possible camera matrices.
    """
    U, S, V = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # create 4 possible camera matrices (Hartley p 258)
    #P = UWV'| +-u3   UW'V'| +-u3
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2s

def skew(x):
    """ Create a skew symmetric matrix *A* from a 3d vector *x*.
        Property: np.cross(A, v) == np.dot(x, v)
    :param x: 3d vector
    :returns: 3 x 3 skew symmetric matrix from *x*
    """
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def reconstruct_one_point(pts1, pts2, m1, m2,intrinsic):
    """
        pt1 and m1 * X are parallel and cross product = 0+++
        pt1 x m1 * X  =  pt2 x m2 * X  =  0
        
        V= [p1 x [I|0]
            P2 X [R|t]]
        V*X=0 => X~ eigenvecor corresponds to the least eignvalue
        
    """
    pts1_e=np.dot(np.linalg.inv(intrinsic),pts1)
    pts2_e=np.dot(np.linalg.inv(intrinsic),pts2)
    num_points = pts1_e.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.vstack([
            np.dot(skew(pts1_e[:,i]), m1),
            np.dot(skew(pts1_e[:,i]), m2)
        ])
        U, S, V = np.linalg.svd(A)
        X = np.ravel(V[-1, :4])
        res[:,i]=X / X[3]
    return res


def linear_triangulation(p1, p2, m1, m2):
    """
    Linear triangulation (Hartley ch 12.2 pg 312) to find the 3D point X
    where p1 = m1 * X and p2 = m2 * X. Solve AX = 0.
    :param p1, p2: 2D points in homo. or catesian coordinates. Shape (3 x n)
    :param m1, m2: Camera matrices associated with p1 and p2. Shape (3 x 4)
    :returns: 4 x n homogenous 3d triangulated points
    """
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (p1[0, i] * m1[2, :] - m1[0, :]),
            (p1[1, i] * m1[2, :] - m1[1, :]),
            (p2[0, i] * m2[2, :] - m2[0, :]),
            (p2[1, i] * m2[2, :] - m2[1, :])
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res


def three_D_reconstruction(Graph, intrinsic,Images):
    """
        pt1 and m1 * X are parallel and cross product = 0+++
        pt1 x m1 * X  =  pt2 x m2 * X  =  0

        V= [p1 x [I|0]
            P2 X [R|t]]
        V*X=0 => X~ eigenvecor corresponds to the least eignvalue

    """
    res=np.array([])
    color_pt=[]
    for j,edge in enumerate(Graph.Edges):
        A =np.array([])
        inxFeature = edge['featureInx']
        viewID = edge['viewID']
        #try:
        color=[]
        for i in range(len(viewID)):

            pt=np.array([0,0,1.0])

            pt[0],pt[1]=Graph.Nodes[viewID[i]]['kp'][inxFeature[i]].pt

            color.append(Images[viewID[i]][int(pt[1]),int(pt[0]),:])
            P=Graph.Nodes[viewID[i]]['P']

            pt_e = np.dot(np.linalg.inv(intrinsic), pt)
            if(A.shape[0]==0):
                A=np.dot(skew(pt_e), P1)
            else:
                A=np.concatenate((A ,np.dot(skew(pt_e), P)),axis=0)


        U, S, V = np.linalg.svd(A)
        X = np.ravel(V[-1, :4])
        color_pt.append(np.mean(color,0))
        if (res.shape[0] == 0):
            res =np.array([X / X[3]])
        else:
            res=np.concatenate((res ,np.array([X / X[3]])),axis=0)
        #except:
        #    pass
    color_pt=np.uint32(np.array(color_pt))
    return res[:,:-1].T,color_pt

def three_D_reconstruction1(Graph,intrinsic):

    X=[]
    res = np.array([])
    W = np.array([])
    for j, edge in enumerate(Graph.Edges):
        W1 = np.array([])
        inxFeature = edge['featureInx']
        viewID = edge['viewID']
        try:
            for i in range(len(viewID)):

                pt = np.array([0, 0, 1.0])

                pt[0], pt[1] = Graph.Nodes[viewID[i]]['kp'][inxFeature[i]].pt
                P1 = Graph.Nodes[viewID[i]]['P']

                #pt_e = np.dot(np.linalg.inv(intrinsic), pt)
                if (W1.shape[0] == 0):
                    W1 = pt
                else:
                    W1 = np.hstack((W1, pt))
        except:
            pass

    W_old = np.zeros_like(W)
    # for i in range(10):
    err = np.sum((W_old - W) ** 2)
    while (err > 1e-3):
        W_old = W
        [U, D, V] = np.linalg.svd(W)
        # print(D)
        P = U[:, :4] @ np.diag(np.sqrt(D[:4]))
        X = np.diag(np.sqrt(D[:4])) @ V[:4, :]
        W = P @ X
        err = np.sum((W_old - W) ** 2)

    return X
    #pts1=np.column_stack((pts1,np.ones((pts1.shape[0]))))
    #pts2=np.column_stack((pts2,np.ones((pts2.shape[0]))))

    
    #pts1=np.int32(pts1)
    #pts2=np.int32(pts2)


    # W=np.hstack((pts1,pts2)).T
    # W_old=np.zeros_like(W)
    # #for i in range(10):
    # err=np.sum((W_old-W)**2)
    # while(err>1e-3):
    #     W_old=W
    #     [U,D,V]=np.linalg.svd(W)
    #     #print(D)
    #     P=U[:,:4]@np.diag(np.sqrt(D[:4]))
    #     X=np.diag(np.sqrt(D[:4]))@V[:4,:]
    #     W=P@X
    #     err=np.sum((W_old - W) ** 2)
    

    
#drawing its lines on the left image
def Estimate_camera_position(pts1,pts2,P1,intrinsic):

    pts1=np.column_stack((pts1,np.ones((pts1.shape[0]))))
    pts2=np.column_stack((pts2,np.ones((pts2.shape[0]))))
    

    pts1_e=np.dot(np.linalg.inv(intrinsic),pts1.T).T
    pts2_e=np.dot(np.linalg.inv(intrinsic),pts2.T).T
    
    
    
    F,mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    E,mask = cv2.findFundamentalMat(pts1_e,pts2_e,cv2.FM_LMEDS)
    
    
    # lines1 = cv2.computeCorrespondEpilines(pts2[:,:-1].reshape(-1,1,2),2,F)
    # lines1=lines1.reshape(-1,3)
    # img5,img6 = drawlines(img1,img2,lines1,pts1[:,:-1],pts2[:,:-1])
    #
    # #drawing its lines on the left image
    # lines2 = cv2.computeCorrespondEpilines(pts1[:,:-1].reshape(-1,1,2),1,F)
    # lines2=lines2.reshape(-1,3)
    # img3,img4 = drawlines(img2,img1,lines2,pts2[:,:-1],pts1[:,:-1])
    #
    # #cv2.imshow('img5',img5)
    # #cv2.imshow('img3',img3)
    
    
    P1_t = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    
    P2s=compute_P_from_essential(E)   
    ind = -1
    for i, P2 in enumerate(P2s):
        # Find the correct camera parameters
        X = reconstruct_one_point(pts1[0, :].reshape(-1,1), pts2[0, :].reshape(-1,1), P1_t, P2,intrinsic)
    
        
        #X = reconstruct_one_point(pts1_e[0:10, :].T, pts2_e[0:10, :].T, P1_t, P2)
    
        # Convert P2 from camera view to world view
        #P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        #d2 = np.dot(P2_homogenous[:3, :4], d1)

        d2=np.dot(P2,X)
    
        if (X[2,:] > 0).all() and (d2[2,:] > 0).all():
            ind = i

    prevOrientation=P1[:,:-1]
    prevLocation=P1[:,-1]
    t=prevLocation+P2s[ind][:,-1]@prevOrientation
    t=np.reshape(t,(-1,1))
    P2=P2s[ind][:,:-1]@prevOrientation
    
    P2=np.hstack((P2,t))
    

    return P2




path='C:/Users/RotemGoren/Downloads/images/'

caliberation_matrix_file_name='calib_parameter.npy'
if os.path.isfile(caliberation_matrix_file_name):
    mtx,dist=np.load(caliberation_matrix_file_name)
    

    
img1=cv2.imread(path+'viff.000.ppm',cv2.COLOR_BGR2RGB)
#img2=cv2.imread(path+'viff.001.ppm',cv2.COLOR_BGR2RGB)


P=[]


#cv2.imshow('img1',img1)
#cv2.imshow('img2',img2)

width,height,_=img1.shape
mtx= np.array([  # for dino
        [2360, 0, width / 2],
        [0, 2360, height / 2],
        [0, 0, 1]])
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

#point3d=np.array([[0,0,0,0]]).T

Graph=Graph()

img = glob.glob(os.path.join(path,"*.ppm"))[0]
prevImage = cv2.imread(img,cv2.COLOR_BGR2RGB)
#sift=cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()
prev_kp, prev_descriptor = sift.detectAndCompute(prevImage, None)
Graph.updateNodes(prev_kp,prev_descriptor,P1)
Images=[]
Images.append(prevImage)

for i,img in enumerate(glob.glob(os.path.join(path,"*.ppm"))[1:3]):

    currImage = cv2.imread(img, cv2.COLOR_BGR2RGB)
    Images.append(currImage)
    sift = cv2.xfeatures2d.SIFT_create()
    curr_kp, curr_descriptor = sift.detectAndCompute(currImage, None)

    matchedPoint1, matchedPoint2,newEdges=find_correspondence_points(prev_kp, prev_descriptor, curr_kp, curr_descriptor,i,i+1)


    #for j in range(len(newEdges)):
    #    newEdges[j]['viewID'] =[i ,i+1]

    '''
    fig, ax = plt.subplots(1, 2)
    ax[0].autoscale_view('tight')
    ax[0].imshow(prevImage)
    ax[0].plot(matchedPoint1[:, 0], matchedPoint2[:, 1], 'r.')
    ax[1].autoscale_view('tight')
    ax[1].imshow(currImage)
    ax[1].plot(matchedPoint2[:, 0], matchedPoint2[:, 1], 'r.')
    fig.show()
    '''

    P2 = Estimate_camera_position(matchedPoint1, matchedPoint2, P1, mtx)
    Graph.updateNodes(curr_kp, curr_descriptor,P2)


    #point3d, pts1, pts2, P1 = three_D_reconstruction1(matchedPoint1, matchedPoint2, P1, mtx)


    #matchedPoint1=np.column_stack((matchedPoint1,np.ones((matchedPoint1.shape[0])))).T
    #matchedPoint2=np.column_stack((matchedPoint2,np.ones((matchedPoint2.shape[0])))).T
    #point3d = reconstruct_one_point(matchedPoint1, matchedPoint2, P1, P2, mtx)

    Graph.updateEdges(newEdges)
    prev_descriptor=curr_descriptor
    prev_kp=curr_kp
    prevImage=currImage
#Graph.sortAndRemove()
point3d,color=three_D_reconstruction(Graph ,mtx,Images)
bundle_adjust(Graph,point3d,mtx)


    #point3d=np.hstack((point3d,tripoints3d))
    #point3d1 = np.hstack((point3d, tripoints3d1))

'''
for image in glob.glob(os.path.join(path,"*.ppm"))[7:9]:
    img2=cv2.imread(image,cv2.COLOR_BGR2RGB)
    
    tripoints3d,pts1,pts2,P1=three_D_reconstruction(img1,img2,P1,mtx)
    point3d=np.hstack((point3d,tripoints3d))
    img1=np.copy(img2)
'''
#for i in range(50):
#    point3d=np.delete(point3d,np.argmax(np.linalg.norm(point3d,axis=0)),axis=1)



fig = plt.figure(1)
fig.suptitle('3D reconstructed', fontsize=16)
ax = fig.gca(projection='3d')
#for i in range(pts2.shape[0]):

ax.scatter(point3d[0,:], point3d[1,:], point3d[2,:],c=color/255.0,s=0.5)

#ax.scatter(point3d1[0,1:], point3d1[1,1:], point3d1[2,1:],c='b',s=2)
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.view_init(elev=135, azim=90)
plt.show()
