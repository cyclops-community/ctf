#include "graph_io_aux.h"

static void *Realloc(void *ptr, size_t sz) {
    
    void *lp;
    
    lp = (void *) realloc(ptr, sz);
    if (!lp && sz) {
        fprintf(stderr, "Cannot reallocate to %zu bytes...\n", sz);
        exit(EXIT_FAILURE);
    }
    return lp;
}

static FILE *Fopen(const char *path, const char *mode) {
    
    FILE *fp = NULL;
    fp = fopen(path, mode);
    if (!fp) {
        fprintf(stderr, "Cannot open file %s...\n", path);
        exit(EXIT_FAILURE);
    }
    return fp;
}

static uint64_t getFsize(FILE *fp) {
    
    int64_t rv;
    uint64_t size = 0;
    
    rv = fseek(fp, 0, SEEK_END);
    if (rv != 0) {
        fprintf(stderr, "SEEK END FAILED\n");
        if (ferror(fp)) fprintf(stderr, "FERROR SET\n");
        exit(EXIT_FAILURE);
    }
    
    size = ftell(fp);
    rv = fseek(fp, 0, SEEK_SET);
    
    if (rv != 0) {
        fprintf(stderr, "SEEK SET FAILED\n");
        exit(EXIT_FAILURE);
    }
    
    return size;
}

void process_order3_tensor(char **led, uint64_t ned, uint64_t **edges) {
    int64_t i = 0;
    uint64_t *ed=(uint64_t *)malloc(ned*sizeof(uint64_t)*(4));

    for (i=0; i<ned; i++) {
        uint64_t i, j, k, v;
        sscanf(led[i],"%lu %lu %lu %lu", &i, &j, &k, &v);
        ed[i*4]  = i;
        ed[i*4+1]= j;
        ed[i*4+2]= k;
        ed[i*4+3]= v;
    }
    *edges = ed;
}

uint64_t read_data_mpiio(int myid, int ntask, const char *fpath, uint64_t **edge, char ***led){
    MPI_File fh;
    MPI_Offset filesize;
    MPI_Offset localsize;
    MPI_Offset start,end;
    MPI_Status status;
    char *chunk = NULL;
    int MPI_RESULT = 0;
    int overlap = 100; // define
    int64_t ned = 0;
    int64_t i = 0;
    
    MPI_RESULT = MPI_File_open(MPI_COMM_WORLD,fpath, MPI_MODE_RDONLY, MPI_INFO_NULL,&fh);
    
    /* Get the size of file */
    MPI_File_get_size(fh, &filesize); //return in bytes
    // FIXME: skewed to give lower processor counts more edges, since
    //        smaller node counts contain fewer characters
    localsize = filesize/ntask;
    start = myid * localsize;
    end = start + localsize;
    end +=overlap;
    
    if (myid  == ntask-1) end = filesize;
    localsize = end - start; //OK
    
    chunk = (char*)malloc( (localsize + 1)*sizeof(char));
    MPI_File_read_at_all(fh, start, chunk, localsize, MPI_CHAR, &status);
    chunk[localsize] = '\0';
    
    int64_t locstart=0, locend=localsize;
    if (myid != 0) {
        while(chunk[locstart] != '\n') locstart++;
        locstart++;
    }
    if (myid != ntask-1) {
        locend-=overlap;
        while(chunk[locend] != '\n') locend++;
        locend++;
    }
    localsize = locend-locstart; //OK
    
    char *data = (char *)malloc((localsize+1)*sizeof(char));
    memcpy(data, &(chunk[locstart]), localsize);
    data[localsize] = '\0';
    free(chunk);
    
    //printf("[%d] local chunk = [%ld,%ld) / %ld\n", myid, start+locstart, start+locstart+localsize, filesize);
    for ( i=0; i<localsize; i++){
        if (data[i] == '\n') ned++;
    }
    //printf("[%d] ned= %ld\n",myid, ned);
    
    (*led) = (char **)malloc(ned*sizeof(char *));
    (*led)[0] = strtok(data,"\n");
    
    for ( i=1; i < ned; i++)
        (*led)[i] = strtok(NULL, "\n");
    
    MPI_File_close(&fh);
    
    return ned;
}
