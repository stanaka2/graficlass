#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

void make_directory(char *directory_name)
{
  mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;

  static char cwd_path[512];

  getcwd(cwd_path, sizeof(cwd_path));

  strcat(cwd_path, "/");
  strcat(cwd_path, directory_name);

  mkdir(cwd_path, mode);
}

void remove_file(char *filename) { remove(filename); }

void remove_dir(char *dirname) { remove(dirname); }
