module Jekyll
  class Navigation < Liquid::Tag

    def initialize(tag_name, data, tokens)
      super
      @data = data
    end

    def lookup(context, name)
      lookup = context.registers
      name.split(".").each { |value| lookup = lookup[value] }
      lookup
    end

    def render_section(data, site, n = 1)
      categories = site.categories

      subs = data.select do |key,name|
        key.count('/') == n
      end

      top = data.select do |key,name|
        key.count('/') < n
      end

      els = top.map do |key,name|
        if n > 1 then
          p = "<ul class='nav nav-list nav-depth'>"
        else
          p = ""
        end
        k = key.gsub('/', '-')
        cat_link = false
        arr = categories[key].map.with_index do |page, i|
          if i == 0 && page.data['title'] == name
            cat_link = true
            "<li class='nav-header'><a href='#{site.baseurl}/#{key}/#{k}'>#{name}</a></li>"
          else
            "<li data-order='#{page.data['order']}'><a href='#{site.baseurl}#{page.url}'>#{page.data['title']}</a></li>"
          end
        end

        if !cat_link
          p += "<li class='nav-header'><a>#{name}</a></li>" + arr.join('')
        else
          p += arr.join('')
        end

        rs = subs.select do |item,name|
          item.include? key
        end

        p += render_section(rs, site, n + 1)

        if n > 1 then
          p + "</ul>"
        else
          p
        end
      end

      els.join('')
    end

    def render(context)
      site = context.registers[:site]
      config = site.config
      sections = config['sections']
      render_section(sections, site)
    end
  end
end

Liquid::Template.register_tag('nav', Jekyll::Navigation)
