#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define COORDINATES_IN_LINE 30
#define MAX_ELEMENTS 60000
#define MAX_DIMENSIONS 30

int elements;
int dimensions;
int k_neighbours;


//save here from file
double *data;
double **data_pointers;

//TEMPORARY 
double *data_received;
double **data_received_pointers;
double *data_received_2;

//nearest neighbours
int *k_nearest_neighbours;
int **k_nearest_neighbours_pointers;
//nearest neighbours distances
double *k_nearest_neighbours_distances;
double **k_nearest_neighbours_distances_pointers;

int *indexes;
int *indexes_temp;


void create_arrays(int block);
void free_indexes(void);
void free_data(void);
void free_memory_for_neighbours(void);
void initialize(int block,int rank);
void print_neighbours(int block);
void read_block(int rank,int block);
void KNN_MPI(int block,int rank);
void update(int block);
void create_temp_arrays(int block);
void initialize_temp_arrays(int block);
void test_comparing_with_matlab(void);
void test_with_sequential(void);



int main(int argc,char **argv)
{
    if(argc!=4)
    {
      printf("Arguments must be 4(including the %s)\n",argv[0]);
      printf("Second Argument is the number of elements\n");
      printf("Third Argument is the number of dimensions\n");
      printf("Fourth Argument is the number of nearest neighbours\n");
      exit(1);
    }
    elements=atoi(argv[1]);
    dimensions=atoi(argv[2]);
    k_neighbours=atoi(argv[3]);
    if(elements<0||elements>MAX_ELEMENTS)
    {
      printf("elements must be between 1 and 60000\n");
      exit(1);
    }
    if(dimensions<0||dimensions>MAX_DIMENSIONS)
    {
      printf("Dimensions must be between 1 and 30\n");
      exit(1);
    }
    if(k_neighbours<0||k_neighbours>=elements)
    {
      printf("k nearest neighbours must be LESS than elements\n");
      exit(1);
    }
    double time_start,time_end,knn_start,knn_end,calculation_time;
    calculation_time=0.0;
    int rank,processes,block;
    //START COMMUNICATION
    MPI_Status status[2];
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&processes);
    if(elements%processes!=0)
    {
      printf("error: elements / processes must have modulo 0\n");
      exit(1);
    }
    block = elements/processes;
    create_arrays(block);
    initialize(block,rank);
    create_temp_arrays(block);

    read_block(rank,block);
    initialize_temp_arrays(block);
    //WAIT FOR ALL PROCESSES TO REACH THIS POINT
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0)
    {
      time_start = MPI_Wtime();
      //IF MUTIPLE PROCESSES ARE RUNNING DO THE RING NETWORK
      //BUT IF YOU ARE THE ONLY ONE YOU HAVE ALREADY CALCULATED THE RESULT
      if(processes!=1)
      {
        //RING NETWORK
        MPI_Send(indexes,block,MPI_INT,(rank+1)%processes,rank,MPI_COMM_WORLD);
        MPI_Send(data,block*dimensions,MPI_DOUBLE,(rank+1)%processes,rank,MPI_COMM_WORLD);

        MPI_Recv(indexes_temp,block,MPI_INT,processes-1,processes-1,MPI_COMM_WORLD,&status[0]);
        MPI_Recv(data_received,block*dimensions,MPI_DOUBLE,processes-1,processes-1,MPI_COMM_WORLD,&status[1]);

        knn_start = MPI_Wtime();
        KNN_MPI(block,rank);
        knn_end = MPI_Wtime();
        calculation_time+=(knn_end-knn_start);

        for(int p=0;p<processes-1;p++)
        {
          MPI_Send(indexes_temp,block,MPI_INT,(rank+1)%processes,rank,MPI_COMM_WORLD);
          MPI_Send(data_received,block*dimensions,MPI_DOUBLE,(rank+1)%processes,rank,MPI_COMM_WORLD);

          MPI_Recv(indexes,block,MPI_INT,processes-1,processes-1,MPI_COMM_WORLD,&status[0]);
          MPI_Recv(data_received_2,block*dimensions,MPI_DOUBLE,processes-1,processes-1,MPI_COMM_WORLD,&status[1]);

          knn_start = MPI_Wtime();
          update(block);
          knn_end = MPI_Wtime();
          calculation_time+=(knn_end-knn_start);

          for(int i=0;i<block;i++)
            indexes_temp[i]=indexes[i];
          for(int i=0;i<block*dimensions;i++)
            data_received[i]=data_received_2[i];
        }

        time_end = MPI_Wtime();
        printf("KNN MPI BLOCKING time = %f\n", time_end-time_start);
        printf("KNN MPI BLOCKING calculation time = %f\n",calculation_time);
        printf("KNN MPI BLOCKING communication time = %f\n",time_end-time_start-calculation_time);

        free_indexes();
        free_data();
        
        //CREATING THE FILE IN WHICH WE SAVE THE RESULT
        int temp_neighbour,return_value;
        FILE *output=fopen("result_mpi_blocking.bin","wb");
        if(output==NULL)
        {
          printf("Error opening the file : result_mpi_blocking.bin\n");
          exit(1);
        }
        for(int i=0;i<block*k_neighbours;i++)
        {
          temp_neighbour=k_nearest_neighbours[i];
          return_value=fwrite(&temp_neighbour,sizeof(int),1,output);
          if(return_value!=1)
          {
            printf("Error writing to file : result_mpi_blocking.bin\n");
            exit(1);
          }
        }
        //print_neighbours(block);
        for(int p=0;p<processes-1;p++)
        {
          MPI_Recv(k_nearest_neighbours,block*k_neighbours,MPI_INT,p+1,p+1,MPI_COMM_WORLD,&status[0]);
          //print_neighbours(block);      
          for(int i=0;i<block*k_neighbours;i++)
          {
            temp_neighbour=k_nearest_neighbours[i];
            return_value=fwrite(&temp_neighbour,sizeof(int),1,output);
            if(return_value!=1)
            {
              printf("Error writing to file : result_mpi_blocking.bin\n");
              exit(1);
            }
          }
        }     
        fclose(output);
        free_memory_for_neighbours(); 
        /*
        //TEST THE RESULT EITHER WITH MATLAB
        // OR WITH THE SEQUENTIAL ALGORITHM
        if(dimensions==MAX_DIMENSIONS)  
          test_comparing_with_matlab();  
        else
          test_with_sequential();
        */
      }
      //IF YOU ARE THE ONLY ONE PROCESS
      //YOU TAKE ALL THE ELEMENTS
      //NOTE!!! 1 PROCESS IS THE SAME AS SEQUENTIAL 
      // IN THIS CASE WE DONT CREATE AND TEST ANYTHING
      else
      {
        KNN_MPI(block,rank);
        time_end = MPI_Wtime();
        printf("KNN MPI BLOCKING time = %f\n", time_end-time_start);
        //print_neighbours_single(block);
      } 
    }
    //IF NOT THE MASTER PROCESS
    else
    {
      //printf("i am rank %d\n",rank);
      MPI_Recv(indexes_temp,block,MPI_INT,rank-1,rank-1,MPI_COMM_WORLD,&status[0]);
      MPI_Recv(data_received,block*dimensions,MPI_DOUBLE,rank-1,rank-1,MPI_COMM_WORLD,&status[1]);

      MPI_Send(indexes,block,MPI_INT,(rank+1)%processes,rank,MPI_COMM_WORLD);
      MPI_Send(data,block*dimensions,MPI_DOUBLE,(rank+1)%processes,rank,MPI_COMM_WORLD);

      KNN_MPI(block,rank);
      for(int p=0;p<processes-1;p++)
      {
        MPI_Recv(indexes,block,MPI_INT,rank-1,rank-1,MPI_COMM_WORLD,&status[0]);
        MPI_Recv(data_received_2,block*dimensions,MPI_DOUBLE,rank-1,rank-1,MPI_COMM_WORLD,&status[1]);

        MPI_Send(indexes_temp,block,MPI_INT,(rank+1)%processes,rank,MPI_COMM_WORLD);
        MPI_Send(data_received,block*dimensions,MPI_DOUBLE,(rank+1)%processes,rank,MPI_COMM_WORLD);

        update(block);

        for(int i=0;i<block;i++)
            indexes_temp[i]=indexes[i];
        for(int i=0;i<block*dimensions;i++)
            data_received[i]=data_received_2[i];
      }

      free_indexes();
      free_data();
      //SENDING BACK THE FULLY UPDATED BLOCK IN ORDER TO CREATE THE RESULT FILE
      MPI_Send(k_nearest_neighbours,block*k_neighbours,MPI_INT,0,rank,MPI_COMM_WORLD);
      free_memory_for_neighbours();
           
    }
    MPI_Finalize();
    //END OF COMMUNICATION

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
  input_c=fopen("result_mpi_blocking.bin","rb");
  if(input_c==NULL)
  {
    printf("error opening the file : result_mpi_blocking.bin\n");
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
    printf("result_mpi_blocking.bin and result_matlab.bin Do not have the same size\n");
    exit(1);
  }
  rewind(input_c);
  rewind(input_matlab);
  for(int i=0;i<elements*k_neighbours;i++)
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
   function test_with_sequential(void)
   
   Using this function we test our result with the sequential one as 
   described in the report
   
*/


void test_with_sequential(void)
{
  FILE *input_mpi,*input_sequential;
  int flag=0;
  int mpi_data,sequential_data;
  int return_value_mpi,return_value_sequential;
  input_mpi=fopen("result_mpi_blocking.bin","rb");
  if(input_mpi==NULL)
  {
    printf("error opening the file : result_mpi_blocking.bin\n");
    exit(1);
  }
  input_sequential=fopen("result_sequential.bin","rb");
  if(input_sequential==NULL)
  {
    printf("error opening the file : result_sequentail.bin\n");
    exit(1);
  }
  fseek(input_mpi,0,SEEK_END);
  fseek(input_sequential,0,SEEK_END);
  if(ftell(input_mpi)!=ftell(input_sequential))
  {
    printf("result_mpi_blocking.bin and result_sequential.bin Do not have the same size\n");
    exit(1);
  }
  rewind(input_mpi);
  rewind(input_sequential);
  for(int i=0;i<elements*k_neighbours;i++)
  {
    return_value_mpi=fread(&mpi_data,sizeof(int),1, input_mpi);
    return_value_sequential=fread(&sequential_data,sizeof(int),1, input_sequential);
    if(return_value_mpi!=1||return_value_sequential!=1)
    {
      printf("error reading the binary files\n");
      exit(1);
    }
    if((sequential_data-mpi_data)!=0)
    {
      flag=1;
      break;
    }
  }
  if(flag)
    printf("Test Failed\n");
  else
    printf("Test Passed\n");
  fclose(input_mpi);
  fclose(input_sequential);
}

/* 
   function update(int block)
   
   In this function we update our block after we take other blocks from the ring network
   The Job we do here is the same with the job we do in function KNN_MPI(int block,int rank).
   

*/


void update(int block)
{
   double distance;
   for(int i=0;i<block;i++)
   {
     for(int j=0;j<block;j++)
     {
       distance=0.0;
       for(int x=0;x<dimensions;x++)
       {
         distance+=(data_pointers[i][x]-data_received_pointers[j][x])*(data_pointers[i][x]-data_received_pointers[j][x]);
       }
       int flag=0;
       int position;
       for(int k=0;k<k_neighbours;k++)
       {
         if(distance<=k_nearest_neighbours_distances_pointers[i][k])
         {
           position=k;
           flag=1;
           break;
         }
       }
       if(flag)
       {
         for(int y=k_neighbours-2;y>=position;y--)
         {
           k_nearest_neighbours_distances_pointers[i][y+1]=k_nearest_neighbours_distances_pointers[i][y];
           k_nearest_neighbours_pointers[i][y+1]=k_nearest_neighbours_pointers[i][y];
         }
         k_nearest_neighbours_distances_pointers[i][position]=distance;
         k_nearest_neighbours_pointers[i][position]=indexes_temp[j];
       }
     }
   }


}

/* 
   function KNN_MPI(int block,int rank)
   
   for each combination of spots int the block( not the spot with itself) we calculated the distance 
   by the definition formula Distance= SUM(Dk-Dm)^2.
   After we have calculated the distance we are searching the array k_nearest_neighbours_distances
   using the adress of each row ( we are using the array k_nearest_neighbours_distances_pointers which 
   contains the adresses of all rows ) in order to find if we will put the distance and the neighbour in 
   our block or not.
   If there is a place for these elements, we save the position we found it.Then 
   we move the elements from this position till the end of the row of the arrays
   (both k nearest neighbours and k nearest neighbours distances ) one step right.
   And Finally we put to the position the label of the element and the distance.


*/

void KNN_MPI(int block,int rank)
{
  double distance;
  for(int i=0;i<block;i++)
  {
    for(int j=0;j<block;j++)
    {
      distance=0.0;
      for(int x=0;x<dimensions;x++)
      {
        if(i!=j)
        {
          distance+=(data_pointers[i][x]-data_pointers[j][x])*(data_pointers[i][x]-data_pointers[j][x]);
        }
      }
      if(i!=j)
      {  
        int flag=0;
        int position;
        for(int k=0;k<k_neighbours;k++)
        {
          if(distance<=k_nearest_neighbours_distances_pointers[i][k])
          {
            position=k;
            flag=1;
            break;
          }
        }
        if(flag)
        {
          for(int y=k_neighbours-2;y>=position;y--)
          {
            k_nearest_neighbours_distances_pointers[i][y+1]=k_nearest_neighbours_distances_pointers[i][y];
            k_nearest_neighbours_pointers[i][y+1]=k_nearest_neighbours_pointers[i][y];
          }
          k_nearest_neighbours_distances_pointers[i][position]=distance;
          k_nearest_neighbours_pointers[i][position]=j+(rank*block);
        }
      }
    }
  }        

}

/* 
   function read_block(int rank,int block)

   Each process opens the file corpus.bin and reads its specific block and save it
   to the array data.
   
  
*/

void read_block(int rank,int block)
{
   int return_value;
   FILE *fp=fopen("corpus.bin","rb");
   if(fp==NULL)
   {
     printf("Error opening the file : corpus.bin\n");
     exit(1);
   }
   for(int i=rank*block;i<(rank+1)*block;i++)
   {
     fseek(fp,i*(sizeof(double)*COORDINATES_IN_LINE+1),SEEK_SET);
     return_value=fread(&data[(i-(rank*block))*dimensions],sizeof(double),dimensions,fp);
     if(return_value!=dimensions)
     {
       printf("Error reading from file : corpus.bin\n");
       exit(1);
     }
   }
   fclose(fp);
}

/* 
   function create_arrays(int block)

   in this function we allocate memory for the arrays:

   data : it will contain the data from the block which is read from the 
          file corpus.bin
   
   data_pointers: it has as elements the address of each row of the array data.
                  we use this array in order to make our calculations easier.

   k_nearest_neighbours : array which contains the neighbour's label.
                          We create this array as an 1D array and not as 
                          an 2D array because we want to send it somewhere else.

   k_nearest_neighbours_pointers : the same as data_pointers for the array data.

   k_nearest_neighbours_distances : array which will contain the distance between a spot and
                                    the i neighbour.
                                    We create this array as an 1D array and not as 
                                    an 2D array because we want to send it somewhere else.

   k_nearest_neighbours_distances_pointers : the same as data_pointers for the array data.

   indexes : array which contains the identity of each spot of the block.
             FOR EXAMPLE:
             if we choose 10 elements and 2 processes
             the first block will have spots from 0 to 4 and the second 5 to 9.

   indexes_temp : array which will be used for the communications.
   
  
*/


void create_arrays(int block)
{
   //ARRAY FOR THE DATA OF EACH BLOCK
   data=(double *)malloc(block*dimensions*sizeof(double));
   if(data==NULL)
   {
     printf("Error allocating memory\n");
     exit(1);
   }

   data_pointers=(double **)malloc(block*sizeof(double *));
   if(data_pointers==NULL)
   {
     printf("Error allocating memory\n");
     exit(1);
   } 
  
   //ARRAY WHICH HAS THE NEAREST NEIGHBOURS OF EACH NODE
   k_nearest_neighbours=(int *)malloc(block*k_neighbours*sizeof(int));
   if(k_nearest_neighbours==NULL)
   {
     printf("Error allocating memory\n");
     exit(1);
   }

   k_nearest_neighbours_pointers=(int **)malloc(block*sizeof(int *));
   if(k_nearest_neighbours_pointers==NULL)
   {
     printf("Error allocating memory\n");
     exit(1);
   }
  
   //ARRAY WHICH CONTAINS THE DISTANCES BETWEEN THE NEAREST DISTANCES
   k_nearest_neighbours_distances=(double *)malloc(block*k_neighbours*sizeof(double));
   if(k_nearest_neighbours_distances==NULL)
   {
     printf("Error allocating memory\n");
     exit(1);
   }

   k_nearest_neighbours_distances_pointers=(double **)malloc(block*sizeof(double *));
   if(k_nearest_neighbours_distances_pointers==NULL)
   {
     printf("Error allocating memory\n");
     exit(1);
   }
  
   indexes=(int *)malloc(block*sizeof(int));
   if(indexes==NULL)
   {
     printf("Error allocating memory\n");
     exit(1);
   }

   indexes_temp=(int *)malloc(block*sizeof(int));
   if(indexes_temp==NULL)
   {
     printf("Error allocating memory\n");
     exit(1);
   }
   

}
/* 
   function initialize(int block,int rank)

   we initialize all the arrays we created in the function create_arrays(int block)
   we give to the array k_nearest_neighbours_distances a very big value because of
   the sorting we have to do and we give to the array k_nearest_neighbours the value -1
   because it is a wrong value and with this we will be able to know if we did a mistake.
   
   the array data_pointers has as elements the addresses of the rows of array data

   the array k_nearest_neighbours_pointers has as elements the addresses of the rows of array k_nearest_neighbours

   the array k_nearest_neighbours_distances_pointers has as elements the addresses of the rows of array k_nearest_neighbours_distances
   
  
*/

void initialize(int block,int rank)
{
   for(int i=0;i<block*dimensions;i++)
     data[i]=0.0;

   for(int i=0;i<block*k_neighbours;i++)
   {
     k_nearest_neighbours[i]=-1.0;
     k_nearest_neighbours_distances[i]=INFINITY;
   }
   
   for(int i=0;i<block;i++)
   {
     indexes[i]=i+(rank*block);
     indexes_temp[i]=0;
   }

   for(int i=0;i<block;i++)
   {
     data_pointers[i]=&data[i*dimensions];
     k_nearest_neighbours_pointers[i]=&k_nearest_neighbours[i*k_neighbours];
     k_nearest_neighbours_distances_pointers[i]=&k_nearest_neighbours_distances[i*k_neighbours];
   }   

}
/*
   function print_neighbours(int block)

   using this function we can print a block (either ours or a block we received)

*/
void print_neighbours(int block)
{
  for(int i=0;i<block;i++)
  {
    for(int j=0;j<k_neighbours;j++)
    {
       printf("%d    ",k_nearest_neighbours_pointers[i][j]);
    }
    printf("\n");
   }
}
/*
   function create_temp_arrays(int block)

   creating arrays which are useful for our communications.

*/
void create_temp_arrays(int block)
{
   data_received=(double *)malloc(block*dimensions*sizeof(double));
   if(data_received==NULL)
   {
     printf("Error allocating memory\n");
     exit(1);
   }

   data_received_pointers=(double **)malloc(block*sizeof(double *));
   if(data_received_pointers==NULL)
   {
     printf("Error allocating memory\n");
     exit(1);
   } 
   
   data_received_2=(double *)malloc(block*dimensions*sizeof(double));
   if(data_received_2==NULL)
   {
     printf("Error allocating memory\n");
     exit(1);
   }
   
}

/*
   function initialize_temp_arrays(int block)

   initializing the arrays created above.

*/
void initialize_temp_arrays(int block)
{
  for(int i=0;i<block*dimensions;i++)
  {
     data_received[i]=0.0;
     data_received_2[i]=0.0;
  }

  for(int i=0;i<block;i++)
     data_received_pointers[i]=&data_received[i*dimensions];

}
/*
   function free_memory_for_neighbours(void)

   free memory which was allocated for the neighbours.

*/
void free_memory_for_neighbours(void)
{
  free(k_nearest_neighbours);
  free(k_nearest_neighbours_pointers);
  free(k_nearest_neighbours_distances);
  free(k_nearest_neighbours_distances_pointers);
}
/*
   function free_data(void)

   free memory which was allocated for arrays which was used for
   saving the block of the file and for arrays which was used for
   our communications ( sends and receives).

*/
void free_data(void)
{
  free(data);
  free(data_pointers);
  free(data_received);
  free(data_received_2);
  free(data_received_pointers);
}
/*
   function free_indexes(void)

   free memory which was allocated for arrays which contained the "id" of each
   spot of the block.
*/
void free_indexes(void)
{
  free(indexes);
  free(indexes_temp);
}

