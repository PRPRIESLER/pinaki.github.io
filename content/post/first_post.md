---
title: 'Few Basic ideas of CORE Features in Python'
date: 2020-11-29T05:23:16+05:30
draft: false
author: 'Pinaki Pani'
description: 'Impact of core python best practices that can help for both library code and also stand-alone codes.'
feature_image: 'https://miro.medium.com/max/10496/1*15eRs65KO-FsrHN3QlY68A.jpeg'
tags: ['Python']
---

## **Protocol Based Data Model: -**

---

So, we have Protocol oriented data model functions in Python. When we look at object orientation in python, we have 3 core features to look into:

- The protocol model of python

- The built-in inheritance protocols

- Some caviars around how object orientation works

Few protocols that comes in real handy when we use object orientation (aka Magic Methods/Dunder(double underscored methods): -

\_\_intit\_\_

\_\_repr\_\_

\_\_add\_\_

\_\_len\_\_

\_\_call\_\_

\_\_ and so on \_\_

Although the use of custom made dunders is discouraged so learning their proper use cases is necessary. Unless you want to actually override their functionality and have the proper idea of implementing them as such so that the whole development team is profited by it, I would suggest otherwise. Cases like devs being ignorant or unaware of the changes or even in a flow of coding, its possible that the methods are used for their trivial functionality but the results come as per the versioned custom method which then leads to failure of test cases. Robustness is not guaranteed so identifying these errors are a huge hackle.

## **\_\_name\_\_ == '\_\_main\_\_': -**

---

Very often in Python programming we come across the above line of code within multiple scripts. The main idea behind it is that in case we want to make sure that our code isn't being imported and run. So, if the condition:

\_\_name\_\_ == '\_\_main\_\_'

checks out then it means that this script was run all by itself and wasn't imported.

```python
#1st Page
if __name__ == '__main__':
    print "Run Directly"
else:
    print "Run From Import

#2nd Page
import first module
print "Second Module's Name: {}".format(__name__)
```

If it was then the code outside that condition will run and whatever is inside its scope will not be bothered. Running below codes: 1'st page -- **Run** **Directly** 2'nd page -- **Run From Import Second Module's Name: main ()**

## **Metaclasses: -**

---

ItΓÇÖs not just the object-oriented python model has hooks and protocols-oriented data models, instead, the whole python language is comprised of hooks and protocols all over.

The class creation process itself can be customized by passing the metaclass keyword argument in the class definition line, or by inheriting from an existing class that included such an argument.

Ok so we have a built-in module in python called \_\_build_class\_\_. Well using this option to make sure how the class creation works isn't really the go-to option. So, the option that we should choose is mentioned below

![](/images/metaclasses.png)

Every class in python has a metaclass that they are subclasses of. This metaclass class denotes how our class will be structured i.e. its not interface, instead if we want something checked if that exist in the class or subclass, we can use metaclass. If we want a customization, we make our own metaclass and make our own class its Subclass. And the metaclass that we are going to write is a subclass right under _Type_. This "_Type_" class is the default metaclass for every class out there in python. So, edit the metaclass by making it _Type's_ subclass!

## **@property: -**

---

@property is used to get the value of a private attribute without using any getter methods. We have to put a line @property in front of the method where we return the private variable. To set the value of the private variable, we use @method_name.setter in front of the method. We have to use it as a setter.

```python
class Property:

    def __init__(self, var):
        ## initializing the attribute
        self.a = var

    @property
    def a(self):
        return self.__a

    ## the attribute name and the method name must be same which is used to set the value for the attribute
    @a.setter
    def a(self, var):
        if var > 0 and var % 2 == 0:
        self.__a = var
        else:
        self.__a = 2
    ## creating an object for the class 'Property'
obj = Property(23)
print(obj.a)
```

There are a few more topics that I would love to mention like: generators, decorators, closures, generator expressions etc. Even though they are pretty common implementing in library code and making your custom features is the tricky part. Hopefully I will mention and put some insight on them in my upcoming weeks post. Surely enough the most excited part is my take on metaclasses and context managers will be something I'll be working on for my next post.
