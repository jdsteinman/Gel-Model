import meshio
import numpy as np

#testvertices1surf.txt
#testfaces1surf.txt

def readVertices(fileName):
    fileObj = open(fileName, "r") #opens the file in read mode
    vertices = fileObj.read().splitlines() #puts the file into an array
    for i, coords in enumerate(vertices):
        vertices[i] = coords.split('\t')
    fileObj.close()
    return vertices


#verts = readVertices('jp testing/samplevertices.txt')

def readFaces(facesFileName,verticesFileName):

    fileFaces = open(facesFileName, "r+") #opens the file in read mode
    oldVertices = readVertices(verticesFileName)
    #print(oldVertices)

    faces = fileFaces.read().splitlines() #puts the file into an array
    
    surfaceVerticesConversion = {}
    numVertices = 0
    newVertices = []

    for i, face in enumerate(faces):
        faces[i] = face.split('\t')
        
        for j, faceVert in enumerate(faces[i]):
            #print(faceVert)

            if faceVert in surfaceVerticesConversion:
                faces[i][j] = surfaceVerticesConversion[faceVert]
            else:
                surfaceVerticesConversion[faceVert] = numVertices
                faces[i][j] = numVertices
                newVertices.append(oldVertices[int(faceVert)])
                numVertices += 1

    

    with open('jp testing/newfaces.txt','w') as outfile:
        for face in faces:
            i = 1
            for vert in face:
                if i != 3:
                    outfile.write('{}\t'.format(vert))
                else:
                    outfile.write('{}'.format(vert))
                i+=1
            outfile.write('\n')

    with open('jp testing/newvertices.txt','w') as outfile:
        for coords in newVertices:
            i = 1
            for coord in coords:
                if i != 3:
                    outfile.write('{}\t'.format(coord))
                else:
                    outfile.write('{}'.format(coord))
                i+=1
            outfile.write('\n')


    fileFaces.close()

    print(surfaceVerticesConversion)
    print(newVertices)

    return faces


readFaces("jp testing/samplefaces.txt",'jp testing/samplevertices.txt')

#readFaces('testfaces1surf.txt','testvertices1surf.txt')



#face = '2869    2871    3136'
#verts = face.split('    ')
#for vert in verts:
#    print(vert)
