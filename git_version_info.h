#ifndef GIT_VERSION_INFO_H
#define GIT_VERSION_INFO_H

#ifdef HAVE_GIT
// Git info is in generated .c file
extern char commit_hash[];
extern char source_dir[];
extern char build_dir[];
extern char branch_name[];
extern char user_name[];
extern char user_email[];
extern char build_time[];
extern int uncommitted_changes;
#else
// Git info is not available
static char commit_hash[] = "not available";
static char source_dir[] = "not available";
static char build_dir[] = "not available";
static char branch_name[] = "not available";
static char user_name[] = "not available";
static char user_email[] = "not available";
static char build_time[] = "not available";
static int uncommitted_changes = 1;
#endif

#endif
