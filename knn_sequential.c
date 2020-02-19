#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#define COORDINATES_IN_LINE 30
#define MAX_ELEMENTS 60000
#define MAX_DIMENSIONS 30

struct timeval startwtime, endwtime;
double seq_time;

int spots;
int dimensions;
int k_nearest_neighbours;

int **nearest_neighbours;
double **nearest_neighbours_distances;
double **data;

void creating_arrays(void);
void initialize(void);
void print(void);
void knn_algorithm_sequential(void);
void free_memory(void);
void read_from_file(void);
void create_output_file(void);
void test_comparing_with_matlab(void);

int main(int argc,char **argv)
{
   if(argc!=4)
   {
     printf("Arguments must be 4(including the %s)\n",argv[0]);
     printf("Second Argument is the number of spots\n");
     printf("Third Argument is the number of dimensions\n");
     printf("Fourth Argument is the number of nearest neighbours\n");
     exit(1);
   }
   spots=atoi(argv[1]);
   dimensions=atoi(argv[2]);
   k_nearest_neighbours=atoi(argv[3]);
   if(spots<0||spots>MAX_ELEMENTS)
   {
     printf("Spots must be between 1 and 60000\n");
     exit(1);
   }
   if(dimensions<0||dimensions>MAX_DIMENSIONS)
   {
     printf("Dimensions must be between 1 and 30\n");
     exit(1);
   }
   if(k_nearest_neighbours<0||k_nearest_neighbours>=spots)
   {
     printf("k nearest neighbours must be LESS than Spots\n");
     exit(1);
   }
   creating_arrays();
   initialize();
   read_from_file();

   gettimeofday (&startwtime, NULL);
   knn_algorithm_sequential();
   gettimeofday (&endwtime, NULL);
   seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
   printf("KNN Sequential time = %f\n", seq_time);

   create_output_file();
   /*
   if(dimensions==MAX_DIMENSIONS)
      test_comparing_with_matlab();
   */
   //print();
   free_memory();


}
/* 
   function test_comparing_with_matlab()
 
   We are comparing the binary file that we created by this program with the same file binary 
   file of matlab.
   NOTE!!!
   if you want to check the result , first you have to create the matlab file called result_matlab.bin
   which will contain the results of matlab for the neighbours of each spot.
   if the data of these files are the same then we pass the test otherwise we fail.
   NOTE 2!!!!
   in this function you will the following :
   matlab_data-c_data!=1
   thats why matlab starts counting from 1 to 60000 and C starts counting from 0 to 59999.
   So if we want to chech the neighbours labels the difference between tha matlab file and our
   file must be 1. 
*/

void test_comparing_with_matlab(void)
{
  FILE *input_c,*input_matlab;
  int flag=0;
  int c_data,matlab_data;
  int return_value_c,return_value_matlab;
  input_c=fopen("result_sequential.bin","rb");
  if(input_c==NULL)
  {
    printf("error opening the file : result_sequential.bin\n");
    exit(1);
  }
  input_matlab=fopen("result_matlab.bin","rb");
  if(input_matlab==NULL)
  {
    printf("error opening the file : result_matlab.bin\n");
    exit(1);
  }
  fseek(input_c,0,SEEK_END);
  fseek(input_matlab,0,SEEK_END);
  if(ftell(input_c)!=ftell(input_matlab))
  {
    printf("result_sequential.bin and result_matlab.bin Do not have the same size\n");
    exit(1);
  }
  rewind(input_c);
  rewind(input_matlab);
  for(int i=0;i<spots*k_nearest_neighbours;i++)
  {
    return_value_c=fread(&c_data,sizeof(int),1, input_c);
    return_value_matlab=fread(&matlab_data,sizeof(int),1, input_matlab);
    if(return_value_c!=1||return_value_matlab!=1)
    {
      printf("error reading the binary files\n");
      exit(1);
    }
    if((matlab_data-c_data)!=1)
    {
      flag=1;
      break;
    }
  }
  if(flag)
    printf("Test Failed\n");
  else
    printf("Test Passed\n");
  fclose(input_c);
  fclose(input_matlab);
}

/* 
   function create_output_file()

   we are creating a file named : result_sequential.bin
   in which we save the array nearest_neighbours 
   we will use this for testing the result in comparison with 
   a similar file created by matlab.
   So this file contains the k nearest neighbours of each spot.
*/

void create_output_file(void)
{
   int temp;
   FILE *out=fopen("result_sequential.bin","wb");
   if(out==NULL)
   {
     printf("error opening the file : result_sequential.bin\n");
     exit(1);
   }
   int return_value;
   for(int i=0;i<spots;i++)
   {
     for(int j=0;j<k_nearest_neighbours;j++)
     {
       temp=nearest_neighbours[i][j];
       return_value=fwrite(&temp,sizeof(int),1,out); 
       if(return_value!=1)
       {
         printf("error writing in file : result_sequential.bin\n");
         exit(1);
       }
     }  
   }
   fclose(out);
}
/* 
   function knn_algorithm_sequential()

   In this function we find for each spot its k nearest neighbours.
   For each combination of elements ( not an element with itself) we find  
   the distance using their coordinates and using the formula :
   distance = SUM (Dk-Dm)^2, k=m from 0 till (dimensions-1) .
   After that we are searching the array k nearest neighbours distances
   in order to see if there is a distance which is bigger than the one we
   just calculated.If there is, we save the position we found it.Then 
   we move the elements from this position till the end of the row of the arrays
   (both k nearest neighbours and k nearest neighbours distances ) one step right.
   And Finally we put to the position the label of the element and the distance.

*/

void knn_algorithm_sequential(void)
{
  double distance_square; 
  for(int i=0;i<spots;i++)
  {
    for(int j=0;j<spots;j++)
    { 
      distance_square=0;
      for(int k=0;k<dimensions;k++)
      {
        if(i!=j)
        {
          distance_square+=(data[i][k]-data[j][k])*(data[i][k]-data[j][k]);
        }
      }
      if(i!=j)
      {
        int flag=0;
        int position;
        for(int x=0;x<k_nearest_neighbours;x++)
        {         
          if(distance_square<=nearest_neighbours_distances[i][x])
          {
            position=x;
            flag=1;
            break;
          }
        }
        if(flag)
        {
          for(int y=k_nearest_neighbours-2;y>=position;y--)
          {
            nearest_neighbours_distances[i][y+1]=nearest_neighbours_distances[i][y];
            nearest_neighbours[i][y+1]=nearest_neighbours[i][y];
          }
          nearest_neighbours_distances[i][position]=distance_square;
          nearest_neighbours[i][position]=j;
        }
      }
    }
  }
      

}
/* 
   function read_from_file()

   we are opening and reading the file corpus.bin which contains the 60000 
   elements and the 30 dimensions of each element
   the user chooses how many elements and dimensions he wants.
   We read them and save the to the 2D array data.
   NOTE!!!
   each row of the file corpus.bin has size 30*sizeof(double)+1
   30 for dimensions, sizeof(double) for each dimension and the 1
   is for the \n we wrote in the file when it was created.
*/

void read_from_file(void)
{
  FILE *fp=fopen("corpus.bin","rb");
  if(fp==NULL)
  {
    printf("error opening the file : corpus.bin\n");
    exit(1);
  }
  int return_value;
  for(int i=0;i<spots;i++)
  {
    fseek(fp,i*((sizeof(double)*COORDINATES_IN_LINE)+1),SEEK_SET);
    return_value=fread(data[i], sizeof(double),dimensions, fp);
    if(return_value!=dimensions)
    {
      printf("error reading from file : corpus.bin\n");
      exit(1);
    }
  }
  fclose(fp);

}
/* 
   function creating_arrays()

   we are allocating memory for the following arrays:
   nearest_neighbours, its size is spots*k nearest neighbours
     and contains the labels of the k nearest neighbours for each spot

   nearest_neighbours_distances, same size as above and it contains the
     distances from the k nearest neighbours for each spot
  
   data, its size is spots*dimensions and it contains the data from the file
     corpus.bin
*/

void creating_arrays(void)
{
   nearest_neighbours  = (int **)malloc(spots*sizeof(int *) );
   if(nearest_neighbours==NULL)
   {
      printf("error allocating memory\n");
      exit(1);
   }
   for(int i=0;i<spots;i++)
   {
       nearest_neighbours[i]=(int *)malloc(k_nearest_neighbours*sizeof(int));
       if(nearest_neighbours[i]==NULL)
       {
         printf("error allocating memory\n");
         exit(1);
       }
   }

   nearest_neighbours_distances  = (double **)malloc(spots*sizeof(double *) );
   if(nearest_neighbours_distances==NULL)
   {
      printf("error allocating memory\n");
      exit(1);
   }
   for(int i=0;i<spots;i++)
   {
       nearest_neighbours_distances[i]=(double *)malloc(k_nearest_neighbours*sizeof(double));
       if(nearest_neighbours_distances[i]==NULL)
       {
         printf("error allocating memory\n");
         exit(1);
       }
   }

   data  = (double **)malloc(spots*sizeof(double *) );
   if(data==NULL)
   {
      printf("error allocating memory\n");
      exit(1);
   }
   for(int i=0;i<spots;i++)
   {
       data[i]=(double *)malloc(dimensions*sizeof(double));
       if(data[i]==NULL)
       {
         printf("error allocating memory\n");
         exit(1);
       }
   }
   
}
/* 
   function initialize()

   using this function we give to the distances array the value INF 
   because of the sorting we have to do in order to find the k nearest 
   neighbours
   we give to the neighbours array the value -1 in order to realize if we made
   a mistake while calculating
*/

void initialize(void)
{   
  for(int i=0;i<spots;i++)
      for(int j=0;j<k_nearest_neighbours;j++)
            nearest_neighbours_distances[i][j]=INFINITY;
  
   for(int i=0;i<spots;i++)
      for(int j=0;j<k_nearest_neighbours;j++)
            nearest_neighbours[i][j]=-1;

}

/* 
   function free_memory()

   using this function we free the memory we
   allocated for our calculations
*/

void free_memory(void)
{
   for(int i=0;i<spots;i++)
     free(nearest_neighbours_distances[i]);
   free(nearest_neighbours_distances);

   for(int i=0;i<spots;i++)
     free(nearest_neighbours[i]);
   free(nearest_neighbours);
   
   for(int i=0;i<spots;i++)
     free(data[i]);
   free(data);
  
}
/* 
   function print()

   using this function we print two arrays:
   nearest_neighbours_distances : contains the distances from the k nearest neighbours
   nearest_neighbours : contains the label of the neighbours (from 0 to 59999)
*/

void print(void)
{
   printf("nearest_neighbours_distances\n");
   for(int i=0;i<spots;i++)
   {
      for(int j=0;j<k_nearest_neighbours;j++)
            printf(" %f   ",nearest_neighbours_distances[i][j]);
      printf("\n");
   }
   
   printf("nearest_neighbours\n");
   for(int i=0;i<spots;i++)
   {
      for(int j=0;j<k_nearest_neighbours;j++)
            printf(" %d   ",nearest_neighbours[i][j]);
      printf("\n");
   }
   
}
