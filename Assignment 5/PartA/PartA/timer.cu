/*
 * timer.c
 */

# define TIMER_C

# include <stdio.h>
# include <sys/time.h>
# include "timer.h"


static double start, stop;        /* store the times locally */
static int start_flag, stop_flag; /* flag timer use */


void initialize_timer ( void )
{
    start = 0.0;
    stop  = 0.0;

    start_flag = 0;
    stop_flag  = 0;
}


void reset_timer ( void )
{
    initialize_timer();
}


void start_timer ( void )
{
    struct timeval time;

    if ( start_flag )
	fprintf( stderr, "WARNING: timer already started!\n" );

    start_flag = 1;

    if ( gettimeofday( &time, NULL ) < 0 )
	perror( "start_timer,gettimeofday" );

    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
}


void stop_timer ( void )
{
    struct timeval time;

    if ( !start_flag )
	fprintf( stderr, "WARNING: timer not started!\n" );

    if ( stop_flag )
	fprintf( stderr, "WARNING: timer already stopped!\n" );

    stop_flag = 1;

    if ( gettimeofday( &time, NULL ) < 0 )
	perror( "stop_timer,gettimeofday" );

    stop = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
}


double elapsed_time ( void )
{
    if ( !start_flag || !stop_flag )
	return (-1.0);

    return (stop-start);
}
