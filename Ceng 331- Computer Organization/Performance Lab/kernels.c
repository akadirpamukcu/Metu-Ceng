/********************************************************
 * Kernels to be optimized for the Metu Ceng Performance Lab
 ********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "defs.h"

/* 
 * Please fill in the following team struct 
 */
team_t team = {
    "AEAEAEAEAEAE",                     /* Team name */

    "Abdulkadir Pamukcu",               /* Third member full name (leave blank if none) */
    "e2237774",                        /* Third member id (leave blank if none) */
    "Efekan Cakmak",            /* First member full name */
    "e2309797",                 /* First member id */

    "Abdullah Yildirim",                /* Second member full name (leave blank if none) */
    "e2310621"                         /* Second member id (leave blank if none) */

};

/****************
 * BOKEH KERNEL *
 ****************/

/****************************************************************
 * Various typedefs and helper functions for the bokeh function
 * You may modify these any way you like.                       
 ****************************************************************/

/* A struct used to compute averaged pixel value */
typedef struct {
    int red;
    int green;
    int blue;
    int num;
} pixel_sum;

/* Compute min and max of two integers, respectively */
static int min(int a, int b) { return (a < b ? a : b); }
static int max(int a, int b) { return (a > b ? a : b); }

/* 
 * initialize_pixel_sum - Initializes all fields of sum to 0 
 */
static void initialize_pixel_sum(pixel_sum *sum) 
{
    sum->red = sum->green = sum->blue = 0;
    sum->num = 0;
    return;
}

/* 
 * accumulate_sum - Accumulates field values of p in corresponding 
 * fields of sum 
 */
static void accumulate_sum(pixel_sum *sum, pixel p) 
{
    sum->red += (int) p.red;
    sum->green += (int) p.green;
    sum->blue += (int) p.blue;
    sum->num++;
    return;
}

/* 
 * assign_sum_to_pixel - Computes averaged pixel value in current_pixel 
 */
static void assign_sum_to_pixel(pixel *current_pixel, pixel_sum sum) 
{
    current_pixel->red = (unsigned short) (sum.red/sum.num);
    current_pixel->green = (unsigned short) (sum.green/sum.num);
    current_pixel->blue = (unsigned short) (sum.blue/sum.num);
    return;
}

/* 
 * avg - Returns averaged pixel value at (i,j) 
 */

static pixel avg(int dim, int i, int j, pixel *src) 
{
    int ii, jj;
    pixel_sum sum;
    pixel current_pixel;

    initialize_pixel_sum(&sum);
    for(ii = max(i-1, 0); ii <= min(i+1, dim-1); ii++) 
    for(jj = max(j-1, 0); jj <= min(j+1, dim-1); jj++) 
        accumulate_sum(&sum, src[RIDX(ii, jj, dim)]);

    assign_sum_to_pixel(&current_pixel, sum);
    return current_pixel;
}

/*******************************************************
 * Your different versions of the bokeh kernel go here 
 *******************************************************/

/* 
 * naive_bokeh - The naive baseline version of bokeh effect with filter
 */
char naive_bokeh_descr[] = "naive_bokeh: Naive baseline bokeh with filter";
void naive_bokeh(int dim, pixel *src, short *flt, pixel *dst) {
  
    int i, j;
    for(i = 0; i < dim; i++) {
        for(j = 0; j < dim; j++) {
            if ( !flt[RIDX(i, j, dim)] )
                dst[RIDX(i, j, dim)] = avg(dim, i, j, src);
            else
                dst[RIDX(i, j, dim)] = src[RIDX(i, j, dim)];
        }
    }
}

/* 
 * bokeh - Your current working version of bokeh
 * IMPORTANT: This is the version you will be graded on
*/
char bokeh_descr[] = "bokeh: Current working version";
void bokeh(int dim, pixel *src, short *flt, pixel *dst) 
{
    int i, j, k, red, green, blue;
    
    // (0,0)
	red = src[0].red + src[1].red;
    green = src[0].green + src[1].green;
    blue = src[0].blue + src[1].blue;
    k = dim+1;
    red += src[dim].red + src[k].red;
    green += src[dim].green + src[k].green;
    blue += src[dim].blue + src[k].blue;
    dst[0].red = (unsigned short) (red>>2);
    dst[0].green = (unsigned short) (green>>2);
    dst[0].blue = (unsigned short) (blue>>2);

    // (dim-1,0)
    k = dim*(dim-1);
    red = src[k].red + src[k+1].red;
    green = src[k].green + src[k+1].green;
    blue = src[k].blue + src[k+1].blue;
    k -= dim;
    red += src[k].red + src[k+1].red;
    green += src[k].green + src[k+1].green;
    blue += src[k].blue + src[k+1].blue;
    dst[k+dim].red = (unsigned short) (red>>2);
    dst[k+dim].green = (unsigned short) (green>>2);
    dst[k+dim].blue = (unsigned short) (blue>>2);
    
	
	// (0,dim-1)
	k = dim-1;
    red = src[k].red + src[k-1].red;
    green = src[k].green + src[k-1].green;
    blue = src[k].blue + src[k-1].blue;
    k += dim;
    red += src[k].red + src[k-1].red;
    green += src[k].green + src[k-1].green;
    blue += src[k].blue + src[k-1].blue;
    dst[dim-1].red = (unsigned short) (red>>2);
    dst[dim-1].green = (unsigned short) (green>>2);
    dst[dim-1].blue = (unsigned short) (blue>>2);
	

    // (dim-1,dim-1)
    k = dim*dim-1;
    red = src[k].red + src[k-1].red;
    green = src[k].green + src[k-1].green;
    blue = src[k].blue + src[k-1].blue;
    k -= dim;
    red += src[k].red + src[k-1].red;
    green += src[k].green + src[k-1].green;
    blue += src[k].blue + src[k-1].blue;
    dst[k+dim].red = (unsigned short) (red>>2);
    dst[k+dim].green = (unsigned short) (green>>2);
    dst[k+dim].blue = (unsigned short) (blue>>2);
	

	// ust kenar
	for(i = 1; i<dim-1; i++){
        red = src[i-1].red + src[i].red + src[i+1].red;
        green = src[i-1].green + src[i].green + src[i+1].green;
        blue = src[i-1].blue + src[i].blue + src[i+1].blue;
        red += src[dim+i-1].red + src[dim+i].red + src[dim+i+1].red;
        green += src[dim+i-1].green + src[dim+i].green + src[dim+i+1].green;
        blue += src[dim+i-1].blue + src[dim+i].blue + src[dim+i+1].blue;
        dst[i].red = (unsigned short) (red/6);
        dst[i].green = (unsigned short) (green/6);
        dst[i].blue = (unsigned short) (blue/6);
	}

	// alt kenar
	k = dim*(dim-1);
	for(i = 1; i<dim-1; i++){
	        red = src[k+i-1].red + src[k+i].red + src[k+i+1].red;
	        green = src[k+i-1].green + src[k+i].green + src[k+i+1].green;
	        blue = src[k+i-1].blue + src[k+i].blue + src[k+i+1].blue;
	        k -= dim;
	        red += src[k+i-1].red + src[k+i].red + src[k+i+1].red;
	        green += src[k+i-1].green + src[k+i].green + src[k+i+1].green;
	        blue += src[k+i-1].blue + src[k+i].blue + src[k+i+1].blue;
	        k += dim;
	        dst[k+i].red = (unsigned short) (red/6);
	        dst[k+i].green = (unsigned short) (green/6);
	        dst[k+i].blue = (unsigned short) (blue/6);
		
	}

	// sol kenar
	k = dim;
	for(i = 1; i<dim-1; i++){
	        k -= dim;
	        red = src[k].red + src[k+1].red;
	        green = src[k].green + src[k+1].green;
	        blue = src[k].blue + src[k+1].blue;
	        k += dim;
	        red += src[k].red + src[k+1].red;
	        green += src[k].green + src[k+1].green;
	        blue += src[k].blue + src[k+1].blue;
	        k += dim;
	        red += src[k].red + src[k+1].red;
	        green += src[k].green + src[k+1].green;
	        blue += src[k].blue + src[k+1].blue;
	        dst[k-dim].red = (unsigned short) (red/6);
	        dst[k-dim].green = (unsigned short) (green/6);
	        dst[k-dim].blue = (unsigned short) (blue/6);
		
	}

	// sag kenar
	k = 2*dim-1;
	for(i = 1; i<dim-1; i++){
	        k -= dim;
	        red = src[k].red + src[k-1].red;
	        green = src[k].green + src[k-1].green;
	        blue = src[k].blue + src[k-1].blue;
	        k += dim;
	        red += src[k].red + src[k-1].red;
	        green += src[k].green + src[k-1].green;
	        blue += src[k].blue + src[k-1].blue;
	        k += dim;
	        red += src[k].red + src[k-1].red;
	        green += src[k].green + src[k-1].green;
	        blue += src[k].blue + src[k-1].blue;
	        dst[k-dim].red = (unsigned short) (red/6);
	        dst[k-dim].green = (unsigned short) (green/6);
	        dst[k-dim].blue = (unsigned short) (blue/6);
	}

	k = dim;
	for(i = 1; i < dim-1; i++) {
        for(j = 1; j < dim-1; j+=2){
        	k -= dim-j;
		    red = src[k].red + src[k+1].red;
		    green = src[k].green + src[k+1].green;
		    blue = src[k].blue + src[k+1].blue;
		    k += dim;
		    red += src[k].red + src[k+1].red;
		    green +=  src[k].green + src[k+1].green;
		    blue +=  src[k].blue + src[k+1].blue;
		    k += dim;
		    red += src[k].red + src[k+1].red;
		    green += src[k].green + src[k+1].green;
		    blue += src[k].blue + src[k+1].blue;
		    k -= dim;
		    dst[k].red = (unsigned short) ((red + src[k-1].red  + src[k-dim-1].red + src[k+dim-1].red)/9);
		    dst[k].green = (unsigned short) ((green + src[k-1].green + src[k-dim-1].green + src[k+dim-1].green)/9);
		    dst[k].blue = (unsigned short) ((blue + src[k-1].blue  + src[k-dim-1].blue + src[k+dim-1].blue)/9);

		    dst[k+1].red = (unsigned short) ((red+ src[k+2].red  + src[k-dim+2].red + src[k+dim+2].red)/9);
		    dst[k+1].green = (unsigned short) ((green + src[k+2].green + src[k-dim+2].green + src[k+dim+2].green)/9);
		    dst[k+1].blue = (unsigned short) ((blue + src[k+2].blue  + src[k-dim+2].blue + src[k+dim+2].blue)/9);
		    k -= j;
		}
		k += dim;
    }

    k = dim*dim;
    for(i=0; i<k; i++,flt++, dst++, src++){
    	if(*flt)
    		*dst = *src;
    }

	
}

/*********************************************************************
 * register_bokeh_functions - Register all of your different versions
 *     of the bokeh kernel with the driver by calling the
 *     add_bokeh_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.  
 *********************************************************************/

void register_bokeh_functions() 
{
    add_bokeh_function(&naive_bokeh, naive_bokeh_descr);   
    add_bokeh_function(&bokeh, bokeh_descr);   
    /* ... Register additional test functions here */
}

/***************************
 * HADAMARD PRODUCT KERNEL *
 ***************************/

/******************************************************
 * Your different versions of the hadamard product functions go here
 ******************************************************/

/* 
 * naive_hadamard - The naive baseline version of hadamard product of two matrices
 */
char naive_hadamard_descr[] = "naive_hadamard The naive baseline version of hadamard product of two matrices";
void naive_hadamard(int dim, int *src1, int *src2, int *dst) {
  
    int i, j;

    for(i = 0; i < dim; i++)
        for(j = 0; j < dim; j++) 
            dst[RIDX(i, j, dim)] = src1[RIDX(i, j, dim)] * src2[RIDX(i, j, dim)];
}

/* 
 * hadamard - Your current working version of hadamard product
 * IMPORTANT: This is the version you will be graded on
 */
char hadamard_descr[] = "hadamard: Current working version";
void hadamard(int dim, int *src1, int *src2, int *dst) 
{   int i;
    int len = dim*dim;
    if(dim & 0b11111){
        for(i = 0; i < len; i++){
        	*dst++ = *src1++ * *src2++;
        }
    }
    else{
        for(i = 0; i < len; i+=32){
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
    		*dst++ = *src1++ * *src2++;
        }
    }
}

/*********************************************************************
 * register_hadamard_functions - Register all of your different versions
 *     of the hadamard kernel with the driver by calling the
 *     add_hadamard_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.  
 *********************************************************************/

void register_hadamard_functions() 
{
    add_hadamard_function(&naive_hadamard, naive_hadamard_descr);   
    add_hadamard_function(&hadamard, hadamard_descr);   
    /* ... Register additional test functions here */
}

