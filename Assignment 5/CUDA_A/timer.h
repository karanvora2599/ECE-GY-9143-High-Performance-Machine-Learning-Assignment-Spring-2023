/**
 * timer.h
 * Function declarations for timer.
 *
 * Jason Connor
 * Sanjay Rajopadhye
 */

# ifndef TIMER_H
# define TIMER_H

/* initialize a timer, this must be done before you can use the timer! */
void initialize_timer ( void );

/* clear the stored values of a timer */
void reset_timer ( void );

/* start the timer */
void start_timer ( void );

/* stop the timer */
void stop_timer ( void );

/* return the elapsed time in seconds, returns -1.0 on error */
double elapsed_time ( void );

# endif /* TIMER_H */
