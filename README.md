#louis Blog

###[View Live louis Blog &rarr;](http://kaya33.github.io)

![](http://jianglife.com/img/blog-desktop.png)


### "Page Build Warning" email

These days, some of you must receive a "Page Build Warning" email from github after you commit happily. **Don't Worried!** It just that github changes its build environment.

In this mail, github told us:

> You are attempting to use the 'pygments' highlighter, which is currently unsupported on GitHub Pages. Your site will use 'rouge' for highlighting instead. To suppress this warning, change the 'highlighter' value to 'rouge' in your '_config.yml'.

So, just edit `_config.yml`, find `highlighter: pygments`, change it to `highlighter: rouge` and the warning will be gone.


## Boilerplate (beta)

Want to clone a boilerplate instead of my buzz blog? Here comes this!  

```
$ git clone git@github.com:kaya33/kaya33.github.io.git
```

**[View Boilerplate Here &rarr;](http://jianglife.com/blog-boilerplate/)**

## Thanks

This theme is forked from [IronSummitMedia/startbootstrap-clean-blog-jekyll](https://github.com/IronSummitMedia/startbootstrap-clean-blog-jekyll)  
Thanks Jekyll and Github Pages!
